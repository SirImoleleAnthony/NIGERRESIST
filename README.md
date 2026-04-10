# NaijaResist: Research-Grade AMR Analytics and Evidence Synthesis for Nigeria

[Live App](https://naijaresist.streamlit.app/) | [GitHub Repository](https://github.com/SirImoleleAnthony/NIGERRESIST/)

NaijaResist is a research-oriented Streamlit platform for antimicrobial resistance (AMR) analytics in Nigeria. It integrates epidemiologic visualization, statistical analysis, predictive modeling, uncertainty quantification, and retrieval-augmented synthesis of AMR literature/policy documents.

This README is intentionally focused on the deployment-relevant application surface (the parts used in production on Streamlit Cloud).

## 1. Deployment Scope (What Powers the Live App)

### 1.1 Core Runtime Files

The deployed app is driven by:

- `app.py` (main Streamlit entry point and page routing)
- `utils.py` (plotting/data helper functions)
- `bhm_functions.py` (statistical analysis helpers and generated observations)
- `train_model.py` (model training workflows)
- `make_prediction.py` (single-case predictive inference workflows)
- `conformal_utils.py` (conformal prediction utilities)
- `rag_system.py` (RAG ingestion, retrieval, and chat generation)
- `requirements.txt` (runtime dependency lock)

### 1.2 Deployment Data Assets

The app runtime depends on:

- `amr_nigeria_full_data.csv`
- `cleaned/*_Nigerian_subset_cleaned.csv`
- `bhm_results/` (precomputed Bayesian/statistical artifacts)
- `rag_files/` (indexed research and policy documents)

### 1.3 Excluded from Deployment (Per .gitignore)

The repository ignore policy intentionally excludes local/dev-only artifacts such as:

- secrets and virtual environments (`.env`, `.venv`, `.streamlit/secrets.toml`)
- test and verification scripts (`test_*.py`, `verify_*.py`)
- local caches and vector stores (`rag_cache/`, `chromadb_store/`)
- local orchestration and legacy migration files (e.g., Docker/Weaviate local setup docs and scripts)
- scratch logs and non-production notes

This keeps the production footprint lean and reproducible.

## 2. Research Motivation

AMR surveillance in low-resource environments often suffers from:

- fragmented isolate metadata
- limited uncertainty-aware analytics
- poor linkage between structured surveillance data and unstructured research/policy text

NaijaResist addresses these by combining:

- structured exploratory analysis of isolates and phenotypes
- statistical and causal analysis pathways
- species-antibiotic predictive modeling with conformal reliability outputs
- evidence-grounded natural-language querying over indexed AMR documents and data summaries

## 3. Analytical Components in Production

### 3.1 Exploratory and Stratified Analysis

The deployed interface supports interactive examination of:

- trends by year and antibiotic
- species-level resistance patterns
- stratification by age group, gender, specialty, and source
- MIC/susceptibility distributions

### 3.2 Statistical and Inferential Analysis

The production app includes:

- Bayesian hierarchical outputs from precomputed artifacts in `bhm_results/`
- non-redundant resistance and gene-sharing interpretation utilities
- temporal analysis and forecasting support (Prophet)
- causal analysis pathways (DoWhy)

### 3.3 Predictive Modeling

Available classifiers include:

- Random Forest
- Logistic Regression
- Gradient Boosting
- K-Nearest Neighbors
- Decision Tree
- XGBoost
- CatBoost
- LightGBM

Model outputs include performance metrics, confusion matrices, and feature-importance views where applicable.

### 3.4 Conformal Prediction for Uncertainty

When CREPES is available in runtime, the app reports:

- conformal prediction sets
- class-wise p-values
- confidence-level categories
- prediction set cardinality for uncertainty communication

This supports reliability-aware interpretation beyond single-label predictions.

### 3.5 RAG-Based AMR Evidence Chat

The deployed RAG stack uses:

- OpenAI embeddings (`text-embedding-3-small`)
- FAISS local vector search with cache directory support
- OpenAI chat model (`gpt-4o-mini`)

Design constraints in production:

- responses are grounded in retrieved context
- source references are emitted as `[Source N]`
- unknowns are explicitly stated when evidence is absent

## 4. Reproducible Deployment-Oriented Setup

### 4.1 Local Reproduction of Production Behavior

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set API key using either Streamlit secrets or environment variable:

```env
OPENAI_API_KEY=your-openai-key
```

Run:

```powershell
streamlit run app.py
```

## 5. Research Workflow Using the Live App

1. Use Data Analysis for descriptive and stratified surveillance insights.
2. Use Statistical Analysis to interrogate hierarchical and temporal signals.
3. Train antibiotic-species models and inspect predictive metrics.
4. Enable conformal prediction to quantify uncertainty.
5. Use Chat with Data to synthesize results against indexed literature/policy context.

## 6. Architecture Summary (Production Path)

1. Ingestion Layer
- CSV surveillance data and document corpora (PDF/CSV).

2. Analysis Layer
- Visualization, statistical interpretation, forecasting, and causal pathways.

3. Modeling Layer
- Classical/boosted classifiers with optional conformal calibration.

4. Retrieval Layer
- Chunking, embedding, FAISS retrieval, and grounded LLM answering.

5. Interface Layer
- Streamlit multi-page workflow orchestrated from `app.py`.

## 7. Scientific Use and Limitations

- The platform supports analytic interpretation and hypothesis generation, not standalone clinical decision-making.
- Performance and calibration are dataset-dependent and sensitive to distribution shift.
- Inference quality depends on data completeness, class balance, and temporal/geographic representativeness.
- Causal outputs require explicit assumptions and should be reported with methodological caution.

## 8. Citation Guidance

For manuscripts, policy briefs, or technical reports, document:

- dataset composition and preprocessing choices
- model family and evaluation protocol
- calibration/uncertainty method (conformal prediction)
- RAG grounding constraints and retrieval-backed citation behavior

---

For deployment access and source code:

- Live app: https://naijaresist.streamlit.app/
- Repository: https://github.com/SirImoleleAnthony/NIGERRESIST/
