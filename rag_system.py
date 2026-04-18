"""
RAG System for Interactive Data Chat — LangChain backend
"""

from __future__ import annotations

import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import PyPDF2
from dotenv import load_dotenv

# LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _load_env_file() -> bool:
    """Load .env from the project directory or its parents."""
    current_dir = Path(__file__).parent
    for search_path in [current_dir, current_dir.parent, current_dir.parent.parent]:
        env_path = search_path / ".env"
        if env_path.exists():
            load_dotenv(env_path, verbose=False)
            return True
    load_dotenv(verbose=False)
    return False


_load_env_file()


def _get_api_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _load_env_file()
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found!\n\n"
            "On Streamlit Cloud: add it via App → Settings → Secrets.\n\n"
            "Locally:\n"
            "  Option 1 – Add to .streamlit/secrets.toml:\n"
            "    OPENAI_API_KEY = \"sk-...your-key...\"\n\n"
            "  Option 2 – Create a .env file:\n"
            "    OPENAI_API_KEY=sk-...your-key...\n\n"
            "Get your API key from: https://platform.openai.com/api-keys"
        )
    return api_key


# ---------------------------------------------------------------------------
# Document loading helpers
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)

def _load_pdfs(pdf_folder: str) -> List[Document]:
    docs: List[Document] = []
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        print(f"Warning: PDF folder '{pdf_folder}' not found — skipping.")
        return docs

    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        try:
            with open(pdf_file, "rb") as fh:
                reader = PyPDF2.PdfReader(fh, strict=False)
                if len(reader.pages) == 0:
                    print(f"Warning: {pdf_file.name} has no pages")
                    continue
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if not text or not text.strip():
                            continue
                        chunks = _splitter.split_text(text)
                        for chunk_idx, chunk in enumerate(chunks):
                            if chunk.strip():
                                docs.append(Document(
                                    page_content=chunk,
                                    metadata={
                                        "source": pdf_file.name,
                                        "page": page_num + 1,
                                        "chunk": chunk_idx + 1,
                                        "type": "pdf",
                                    },
                                ))
                    except Exception as page_err:
                        print(f"  Page {page_num + 1} of {pdf_file.name} skipped: {str(page_err)[:80]}")
            if any(d.metadata["source"] == pdf_file.name for d in docs):
                print(f"Loaded {pdf_file.name}")
            else:
                print(f"No extractable text in {pdf_file.name}")
        except Exception as e:
            print(f"Error reading {pdf_file.name}: {type(e).__name__}: {str(e)[:200]}")

    return docs


def _csv_to_documents(df: pd.DataFrame, filename: str) -> List[Document]:
    docs: List[Document] = []
    base_meta = {"source": filename, "type": "csv"}

    # ---- Detect AMR-style dataset ----------------------------------------
    # Known antibiotic columns in the Nigeria dataset
    _AB = [
        "Amikacin", "Cefepime", "Ceftazidime", "Amoxycillin clavulanate",
        "Ampicillin sulbactam", "Ampicillin", "Aztreonam", "Ceftaroline",
        "Ceftazidime avibactam", "Ciprofloxacin", "Colistin", "Gentamicin",
        "Levofloxacin", "Imipenem", "Meropenem", "Piperacillin tazobactam",
        "Tigecycline", "Trimethoprim sulfa",
    ]
    present_ab = [ab for ab in _AB if ab in df.columns]
    interp_cols = [c for c in df.columns if c.endswith("_I") and c[:-2] in _AB]
    is_amr = len(present_ab) >= 3

    if is_amr:
        n = len(df)
        species_col = next((c for c in ["Species", "species", "Organism", "organism"] if c in df.columns), None)
        year_col = next((c for c in ["Year", "year"] if c in df.columns), None)

        # -- Global overview document --------------------------------------
        year_info = ""
        if year_col:
            try:
                yr_min = int(df[year_col].min())
                yr_max = int(df[year_col].max())
                year_info = f"Year range: {yr_min}–{yr_max}. "
            except Exception:
                pass

        species_info = ""
        if species_col:
            top_sp = df[species_col].value_counts().head(8)
            species_info = "Top species: " + "; ".join(
                f"{sp} (n={cnt})" for sp, cnt in top_sp.items()
            ) + ". "

        ab_list = ", ".join(present_ab)
        overview = (
            f"Dataset overview: {filename}\n"
            f"Total isolates: {n}. "
            f"{year_info}"
            f"{species_info}"
            f"Antibiotics tested ({len(present_ab)}): {ab_list}.\n"
            f"Columns: {', '.join(df.columns.tolist())}."
        )
        docs.append(Document(page_content=overview, metadata={**base_meta, "doc_subtype": "overview"}))

        # -- Per-antibiotic resistance summary documents -------------------
        for ab in present_ab:
            i_col = f"{ab}_I"
            if i_col in df.columns:
                counts = df[i_col].value_counts().to_dict()
                total = df[i_col].notna().sum()
                r = counts.get("Resistant", 0)
                s = counts.get("Susceptible", 0)
                i_cnt = counts.get("Intermediate", 0)
                pct_r = f"{100*r/total:.1f}%" if total else "N/A"
                pct_s = f"{100*s/total:.1f}%" if total else "N/A"
                lines = [
                    f"Antibiotic resistance summary: {ab} | Dataset: {filename}",
                    f"Total isolates tested: {total}",
                    f"Resistant: {r} ({pct_r})",
                    f"Susceptible: {s} ({pct_s})",
                    f"Intermediate: {i_cnt}",
                ]
                if species_col:
                    sp_r = (
                        df[df[i_col] == "Resistant"][species_col]
                        .value_counts()
                        .head(5)
                    )
                    if not sp_r.empty:
                        lines.append(
                            "Top resistant species: "
                            + "; ".join(f"{sp} (n={cnt})" for sp, cnt in sp_r.items())
                        )
            else:
                # No interpretation column — just list MIC distribution
                mic_dist = df[ab].value_counts().head(10).to_dict()
                lines = [
                    f"Antibiotic MIC distribution: {ab} | Dataset: {filename}",
                    f"Total measurements: {df[ab].notna().sum()}",
                    "MIC values: " + "; ".join(f"{k}: {v}" for k, v in mic_dist.items()),
                ]
            docs.append(Document(
                page_content="\n".join(lines),
                metadata={**base_meta, "doc_subtype": "antibiotic_summary", "antibiotic": ab},
            ))

        # -- Per-species resistance profile documents ----------------------
        if species_col and interp_cols:
            for sp, grp in df.groupby(species_col):
                if len(grp) < 5:
                    continue
                ab_lines = [f"Species resistance profile: {sp} | Dataset: {filename} | n={len(grp)}"]
                for i_col in interp_cols:
                    ab_name = i_col[:-2]
                    if ab_name not in df.columns:
                        continue
                    c = grp[i_col].value_counts().to_dict()
                    total = sum(c.values())
                    r = c.get("Resistant", 0)
                    pct = f"{100*r/total:.0f}%" if total else "N/A"
                    ab_lines.append(f"  {ab_name}: {r}/{total} resistant ({pct})")
                docs.append(Document(
                    page_content="\n".join(ab_lines),
                    metadata={**base_meta, "doc_subtype": "species_profile", "species": str(sp)},
                ))

        # -- Sample of raw rows for specific record look-ups ---------------
        sample = df.sample(min(200, n), random_state=42) if n > 200 else df
        for idx, row in sample.iterrows():
            content = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            docs.append(Document(
                page_content=content,
                metadata={**base_meta, "doc_subtype": "row", "row": int(idx)},
            ))

    else:
        # Generic CSV — embed raw rows (capped at 500)
        sample = df.sample(min(500, len(df)), random_state=42) if len(df) > 500 else df
        for idx, row in sample.iterrows():
            content = " | ".join(f"{col}: {val}" for col, val in row.items())
            docs.append(Document(
                page_content=content,
                metadata={**base_meta, "row": int(idx)},
            ))

    print(f"  Created {len(docs)} documents from {filename} ({len(df)} rows)")
    return docs


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a seasoned researcher with more than 3 decades of experience specialising in Antimicrobial Resistance (AMR) in Nigeria with extensive knowledge of local healthcare systems, microbial patterns, and public health policies. You also have research experience in AMR in other African countries and globally, allowing you to contextualise Nigerian data within broader trends. You are adept at analysing complex datasets, extracting insights, and communicating findings clearly to both scientific and non-technical audiences.

**YOUR PRIMARY ROLE:**
Analyse and answer questions about AMR using the provided context from documents and data.

**CRITICAL RULES:**
1. ALWAYS use the provided context to answer questions
2. You CANNOT use external knowledge or general training data
3. You can analyse, synthesise, and interpret the provided data
4. You MUST cite your sources using the label provided in the context (e.g. [Source 1], [Source 2])
5. If data is missing that you need, say so clearly

**WHAT YOU CAN DO:**
- Extract and analyse specific data points
- Identify patterns and trends from the provided data
- Compare different entries or time periods from the data
- Synthesise information from multiple sources
- Calculate statistics or summaries from the data
- Answer follow-up questions about the context

**WHAT YOU CANNOT DO:**
- Use medical knowledge not in the context
- Make predictions beyond what the data supports
- Suggest treatments or clinical recommendations
- Reference external studies or sources

**IF YOU DON'T HAVE DATA:**
Say "I don't have data about [topic]" \u2014 do NOT refuse to analyse what IS available.

**RESPONSE STYLE:**
1. Answer directly with findings from the data
2. Show the data support (cite sources and key values)
3. Be honest about gaps
4. Explain what the data shows, even if incomplete"""


# ---------------------------------------------------------------------------
# RAG System -- LangChain + FAISS backend
# ---------------------------------------------------------------------------

class RAGSystem:

    CACHE_DIR = "rag_cache"
    FAISS_INDEX_DIR = os.path.join("rag_cache", "faiss_lc")
    META_FILE = os.path.join("rag_cache", "pdf_meta.pkl")

    def __init__(self) -> None:
        self.documents: List[Dict] = []          # legacy-compat dict list for app.py
        self._lc_docs: List[Document] = []       # LangChain Document objects
        self._vectorstore: Optional[FAISS] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._llm: Optional[ChatOpenAI] = None
        self.conversation_history: List[Dict] = []
        self.csv_uploaded: bool = False

    # ------------------------------------------------------------------
    # Lazy LangChain object factories
    # ------------------------------------------------------------------

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=_get_api_key(),
            )
        return self._embeddings

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model="gpt-5.4-mini",
                temperature=0.2,
                max_tokens=1500,
                openai_api_key=_get_api_key(),
            )
        return self._llm

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def load_base_documents(self, pdf_folder: str) -> None:
        pdf_lc = _load_pdfs(pdf_folder)
        self._lc_docs.extend(pdf_lc)
        self.documents.extend(
            {**doc.metadata, "content": doc.page_content} for doc in pdf_lc
        )
        print(f"Loaded {len(pdf_lc)} PDF chunks")

    def load_csv_file(self, csv_path: str) -> None:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV {csv_path}: {e}")
            return
        filename = Path(csv_path).name
        csv_lc = _csv_to_documents(df, filename)
        self._lc_docs.extend(csv_lc)
        self.documents.extend(
            {**doc.metadata, "content": doc.page_content} for doc in csv_lc
        )
        print(f"Loaded {len(csv_lc)} documents from {filename}")

    def reload_base_documents(self, pdf_folder: str) -> None:
        csv_lc = [d for d in self._lc_docs if d.metadata.get("type") == "csv"]
        csv_dict = [d for d in self.documents if d.get("type") == "csv"]
        self._lc_docs = csv_lc
        self.documents = csv_dict
        pdf_lc = _load_pdfs(pdf_folder)
        self._lc_docs.extend(pdf_lc)
        self.documents.extend(
            {**doc.metadata, "content": doc.page_content} for doc in pdf_lc
        )
        print(f"Reloaded {len(pdf_lc)} PDF chunks.  Total: {len(self.documents)}")

    def add_user_csv(self, csv_bytes: bytes, filename: str) -> None:
        try:
            df = pd.read_csv(BytesIO(csv_bytes))
        except Exception as e:
            print(f"Error reading CSV {filename}: {e}")
            return

        print(f"Processing {filename} ({len(df)} rows) …")
        csv_lc = _csv_to_documents(df, filename)
        if not csv_lc:
            print(f"No documents created from {filename}")
            return

        self._lc_docs.extend(csv_lc)
        self.documents.extend(
            {**doc.metadata, "content": doc.page_content} for doc in csv_lc
        )

        embeddings = self._get_embeddings()
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(csv_lc, embeddings)
        else:
            self._vectorstore.add_documents(csv_lc)

        self.csv_uploaded = True
        print(f"Indexed {len(csv_lc)} summary/row documents from {filename}")

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------

    def build_vector_store(
        self,
        show_progress: bool = False,
        force_rebuild: bool = False,
    ) -> bool:
        # All docs loaded from disk (PDFs or pre-loaded CSVs): persisted to cache
        base_lc = [d for d in self._lc_docs if d.metadata.get("type") in ("pdf", "csv")]
        # User-uploaded CSVs that arrived after initialization (runtime only)
        user_lc: List[Document] = []   # not used here; added via add_user_csv

        if not base_lc:
            print("No documents to index")
            return False

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        embeddings = self._get_embeddings()

        # --- Try loading from disk cache ----------------------------------
        if not force_rebuild:
            try:
                if (
                    os.path.exists(self.FAISS_INDEX_DIR)
                    and os.path.exists(self.META_FILE)
                ):
                    with open(self.META_FILE, "rb") as fh:
                        cached_count: int = pickle.load(fh)

                    if cached_count == len(base_lc):
                        vs = FAISS.load_local(
                            self.FAISS_INDEX_DIR,
                            embeddings,
                            allow_dangerous_deserialization=True,
                        )
                        self._vectorstore = vs
                        print(
                            f"Loaded FAISS index from cache "
                            f"({vs.index.ntotal} vectors)"
                        )
                        return True
            except Exception as e:
                print(f"Cache load failed ({e}) — rebuilding from scratch.")

        # --- Build from scratch -------------------------------------------
        print(f"Indexing {len(base_lc)} document chunks via OpenAI Embeddings …")
        print("  (This runs once; future starts load from the disk cache.)")

        if show_progress:
            try:
                import streamlit as st
                st.info(f"Generating embeddings for {len(base_lc)} chunks …")
            except Exception:
                pass

        try:
            self._vectorstore = FAISS.from_documents(base_lc, embeddings)
        except Exception as e:
            print(f"Failed to build vector store: {e}")
            return False

        # --- Persist index to disk ----------------------------------------
        try:
            self._vectorstore.save_local(self.FAISS_INDEX_DIR)
            with open(self.META_FILE, "wb") as fh:
                pickle.dump(len(base_lc), fh)
            print(f"Saved FAISS index to {self.FAISS_INDEX_DIR}/")
        except Exception as e:
            print(f"Could not save index cache: {e}")
        except Exception as e:
            print(f"Could not save index cache: {e}")

        print(f"FAISS index built: {self._vectorstore.index.ntotal} vectors")

        if show_progress:
            try:
                import streamlit as st
                st.success("RAG system ready!")
            except Exception:
                pass

        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        if self._vectorstore is None:
            return []

        try:
            results = self._vectorstore.similarity_search_with_score(query, k=k * 2)
        except Exception as e:
            print(f"Search error: {e}")
            return []

        retrieved: List[Dict] = []
        for doc, score in results:
            entry = {**doc.metadata, "content": doc.page_content, "relevance_score": float(score)}
            retrieved.append(entry)

        # Lower L2 score = more similar; sort ascending
        retrieved.sort(key=lambda x: x["relevance_score"])

        if self.csv_uploaded:
            csv_hits = [d for d in retrieved if d.get("type") == "csv"]
            pdf_hits = [d for d in retrieved if d.get("type") == "pdf"]
            retrieved = csv_hits + pdf_hits

        # Attach a sequential label so the LLM can cite "[Source N]"
        for i, doc in enumerate(retrieved[:k], 1):
            doc["source_label"] = f"Source {i}"

        return retrieved[:k]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> Tuple[str, List[Dict]]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        self.conversation_history.append({"role": "user", "content": user_message})

        relevant_docs = self.search_documents(user_message, k=8)

        if not relevant_docs:
            no_ctx = (
                f"I don't have relevant information to answer: \"{user_message}\"\n\n"
                "My knowledge base currently contains:\n"
                "- Research papers on AMR in Nigeria\n"
                "- Any CSV files you have uploaded this session\n\n"
                "Try asking about:\n"
                "  • Specific antibiotics (e.g. Ciprofloxacin, Ampicillin)\n"
                "  • Bacterial species or resistance patterns\n"
                "  • Data trends or comparisons"
            )
            self.conversation_history.append({"role": "assistant", "content": no_ctx})
            return no_ctx, []

        context_block = self._build_context(relevant_docs)

        messages = [SystemMessage(content=_SYSTEM_PROMPT)]
        for msg in self.conversation_history[-7:-1]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        user_content = (
            f"{context_block}\n\n"
            "---USER QUESTION---\n"
            f"{user_message}\n\n"
            "---INSTRUCTIONS---\n"
            "Analyse the provided context to answer the question.\n"
            "Use data values, statistics, and patterns from the context.\n"
            "Cite sources using their [Source N] label only (e.g. [Source 1], [Source 2]).\n"
            "Be direct and clear with findings.\n"
            "Do NOT use knowledge outside the context.\n"
            "If data is incomplete, explain what you can see and what is missing."
        )
        messages.append(HumanMessage(content=user_content))

        try:
            response = self._get_llm().invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"Error generating response: {e}"
            print(answer)

        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer, relevant_docs

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, documents: List[Dict]) -> str:
        if not documents:
            return "NO RELEVANT DOCUMENTS FOUND IN DATABASE"

        sep = "=" * 70
        ctx = (
            "RETRIEVED CONTEXT FROM DATABASE:\n"
            "The following is ALL the information available to answer your question.\n"
            "Use ONLY this information.  Do NOT use external knowledge.\n"
            "When citing information, refer to it ONLY by its label, e.g. [Source 1].\n"
            "Do NOT mention filenames, page numbers, or chunk numbers in your response.\n"
            f"{sep}\n\n"
        )

        pdf_docs = [d for d in documents if d.get("type") == "pdf"]
        csv_docs = [d for d in documents if d.get("type") == "csv"]

        if pdf_docs:
            ctx += "FROM RESEARCH PAPERS & DOCUMENTS:\n"
            for doc in pdf_docs:
                label = doc.get("source_label", "Source ?")
                ctx += (
                    f"\n[{label}]\n"
                    f"    Content: {doc.get('content', '')[:700]}\n"
                )

        if csv_docs:
            if pdf_docs:
                ctx += f"\n{sep}\n"
            ctx += "FROM RESISTANCE DATA:\n"
            for doc in csv_docs:
                label = doc.get("source_label", "Source ?")
                ctx += (
                    f"\n[{label}]\n"
                    f"    Content: {doc.get('content', '')[:700]}\n"
                )

        ctx += (
            f"\n{sep}\n"
            "IMPORTANT: Your answer MUST come from the above context only.\n"
            "Cite sources using [Source N] labels only. Do NOT reveal filenames or page numbers.\n"
            "If something is not in the context, say it's not available.\n"
        )
        return ctx

    # ------------------------------------------------------------------
    # Helpers (public API used by app.py)
    # ------------------------------------------------------------------

    def _get_document_type_summary(self) -> str:
        if not self.documents:
            return "No documents"
        counts: Dict[str, int] = {}
        for doc in self.documents:
            key = f"{doc.get('type', 'unknown')}: {doc.get('source', 'unknown')}"
            counts[key] = counts.get(key, 0) + 1
        return " | ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history

    def clear_history(self) -> None:
        self.conversation_history = []

    def reset_with_new_data(self, csv_bytes: bytes, filename: str) -> None:
        self.clear_history()
        self.add_user_csv(csv_bytes, filename)


# ---------------------------------------------------------------------------
# Public initialisation helper (same signature as before)
# ---------------------------------------------------------------------------

def initialize_rag_system(
    pdf_folder: str = "rag_files",
    csv_path: Optional[str] = None,
    show_progress: bool = False,
    force_rebuild: bool = False,
) -> RAGSystem:
    rag = RAGSystem()
    rag.load_base_documents(pdf_folder)

    # Auto-load the main CSV dataset if provided
    if csv_path and Path(csv_path).exists():
        rag.load_csv_file(csv_path)
    elif csv_path:
        print(f"Warning: csv_path '{csv_path}' not found — skipping.")

    rag.build_vector_store(show_progress=show_progress, force_rebuild=force_rebuild)
    return rag
