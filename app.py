import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from plotly import graph_objs as go
import os
from pathlib import Path


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
from dowhy import CausalModel
import networkx as nx
import numpy as np
import pickle
import time
from io import BytesIO
from utils import (
    plot_trend, plot_by_species, plot_by_age, plot_by_gender,
    plot_by_specialty, plot_by_source, plot_by_gene,
    plot_mic_distribution, plot_species_by_gender, plot_species_by_age,
    plot_species_over_years, plot_species_by_susceptibility,
    plot_mic_by_species, plot_species_trend_by_susceptibility, load_data
)
from bhm_functions import (
    get_available_antibiotics, load_nrrs_analysis, generate_nrrs_observation,
    load_gene_sharing_analysis, generate_hgt_observation,
    load_temporal_analysis, generate_temporal_observation
)
from train_model import train_model
from make_prediction import make_prediction
from rag_system import RAGSystem, initialize_rag_system


#-----------Web page setting-------------------#
page_title = "Nigeria AMR Hub"
page_icon = "🦠🧬💊"
picker_icon = "👇"
#layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = "wide")

#--------------------RAG System Initialization------------------#
# NOTE: @st.cache_resource holds a global lock while running, which blocks
# every other browser tab until it completes.  We intentionally do NOT use it
# here so the function is kept lightweight and called lazily per session via
# st.session_state.  The FAISS-based RAGSystem caches its index to disk so
# repeated calls still return quickly after the first build.
def load_rag_system_cached():
    try:
        start_time = time.time()
        print("[RAG] Initialising RAG system (FAISS backend)...")
        rag = initialize_rag_system(
            pdf_folder="rag_files",
            csv_path="amr_nigeria_full_data.csv",
            show_progress=True,
        )
        elapsed = time.time() - start_time
        n_docs = len(rag.documents) if rag else 0
        print(f"[RAG] Ready in {elapsed:.1f}s — {n_docs} documents indexed")
        return rag
    except ValueError as e:
        # Typically missing API key — surface to the user
        print(f"[RAG] Configuration error: {e}")
        raise
    except Exception as e:
        print(f"[RAG] Error initialising: {e}")
        import traceback
        print(traceback.format_exc())
        return None

#--------------------Web App Design----------------------#

selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'Data Analysis', 'Statistical Analysis', 'Train Model', 'Make Prediction', 'Chat with Data', 'About'],
    icons = ["house-fill", "bar-chart-fill", "graph-up", "cpu-fill", "robot", "chat-dots-fill", "info-circle-fill"],
    default_index = 0,
    orientation = "horizontal"
)

# Data is loaded lazily per page to avoid errors if files are missing

if selected == 'Home':
    st.title("Welcome to the Nigeria AMR Hub! " + page_icon)
    st.markdown("""
    The Nigeria AMR Hub is a comprehensive platform dedicated to addressing the challenges of antimicrobial resistance (AMR) in Nigeria. Our mission is to provide a centralized resource for researchers, healthcare professionals, policymakers, and the general public to access data, insights, and tools related to AMR in Nigeria. 

    **Key Features:**
    - **Data Repository:** Access a wide range of datasets related to AMR in Nigeria, including surveillance data, research studies, and clinical reports.
    - **Data Analysis Tools:** Utilize built-in tools for analyzing AMR data, identifying trends, and generating visualizations.
    - **Statistical Analysis:** Perform advanced statistical analyses to understand the factors contributing to AMR and evaluate intervention strategies.
    - **Machine Learning Models:** Train and deploy machine learning models to predict AMR patterns and inform decision-making.
    - **Community Engagement:** Connect with experts, share insights, and collaborate on initiatives to combat AMR in Nigeria.

    Join us in our efforts to combat antimicrobial resistance and improve public health outcomes in Nigeria!
    """)


elif selected == 'Chat with Data':
    st.title("Chat with Data " + page_icon)
    st.markdown("""
    Engage in interactive conversations with our AMR data and research papers! Ask questions, explore trends, and gain insights from our comprehensive datasets on antimicrobial resistance (AMR) in Nigeria. Our AI assistant will analyze both the official data and any custom data you upload to provide comprehensive insights.
    """)
    
    # Initialize RAG system lazily on first visit to this page
    if "rag_system" not in st.session_state:
        if os.path.exists("rag_files"):
            rag_load_error = None
            with st.spinner("Loading RAG system (first run builds a local index — may take a minute)..."):
                try:
                    rag = load_rag_system_cached()
                    st.session_state.rag_system = rag
                except ValueError as api_err:
                    # Missing API key — surface a clear message
                    rag_load_error = str(api_err)
                    st.session_state.rag_system = None
                    st.session_state.rag_load_error = rag_load_error
                except Exception as e:
                    rag_load_error = str(e)
                    st.session_state.rag_system = None
                    st.session_state.rag_load_error = rag_load_error
        else:
            st.session_state.rag_system = None
            st.session_state.rag_load_error = "`rag_files` folder not found."

    # Check if RAG system loaded successfully
    if st.session_state.get("rag_system") is None:
        load_error = st.session_state.get("rag_load_error", "Unknown error during initialisation.")
        st.error(f"RAG system failed to load:\n\n{load_error}")

        # Offer retry
        if st.button("🔄 Try Again"):
            # Clear cached state so the block above runs again
            for key in ("rag_system", "rag_load_error"):
                st.session_state.pop(key, None)
            st.rerun()
        st.stop()
    
    # Initialize chat history and uploaded files on first visit to this page
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### Chat Settings")
        
        # Display available documents - dynamically read from rag_files folder
        with st.expander("📚 Available Documents"):
            # Get list of PDFs from rag_files folder
            rag_folder = Path("rag_files")
            pdf_files = sorted(rag_folder.glob("*.pdf")) if rag_folder.exists() else []
            
            if pdf_files:
                st.markdown("**Research Papers & Resources:**")
                for pdf in pdf_files:
                    # Clean up filename for display
                    display_name = pdf.stem.replace("_", " ")
                    st.write(f"• {display_name}")
            
            st.markdown("\n**Data:**")
            st.write("• Nigeria AMR Full Dataset (20,000+ records)")
            st.write("• Additional CSV files you upload")
        
        # Show indexed documents breakdown
        with st.expander("📊 Indexed Documents Breakdown"):
            if st.session_state.rag_system:
                doc_summary = st.session_state.rag_system._get_document_type_summary()
                st.info(f"**Total: {len(st.session_state.rag_system.documents)} documents**\n\n{doc_summary}")
                
                # Add rebuild option
                if st.button("🔄 Rebuild Vector Store (if PDFs changed)", use_container_width=True):
                    with st.spinner("🔄 Rebuilding vector store with new PDFs..."):
                        try:
                            # Force reload and rebuild of the vector store
                            st.session_state.rag_system.reload_base_documents("rag_files")
                            st.session_state.rag_system.build_vector_store(force_rebuild=True)
                            st.success("✅ Vector store rebuilt successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error rebuilding: {str(e)}")
            else:
                st.warning("RAG system not loaded")
        
        # Upload custom CSV
        st.markdown("### Upload Custom Data")
        st.caption("Add your AMR data (CSV) to search and analyze together with our PDFs")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with your AMR data",
            type="csv",
            help="Your data will be indexed and prioritized in search results",
            key=f"csv_uploader_{len(st.session_state.uploaded_files)}"
        )
        
        if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"🔄 Indexing {uploaded_file.name}..."):
                try:
                    csv_bytes = uploaded_file.read()
                    st.session_state.rag_system.add_user_csv(csv_bytes, uploaded_file.name)
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} added! Your data is now prioritized in searches.")
                except Exception as e:
                    st.error(f"❌ Error uploading file: {str(e)}")
        
        if st.session_state.uploaded_files:
            st.markdown("**Uploaded Files:**")
            for file in st.session_state.uploaded_files:
                st.write(f"• {file}")
        
        # New chat button
        if st.button("🔄 Start New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag_system.clear_history()
            st.session_state.uploaded_files = []
            st.rerun()
    
    # Main chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar="🦠"):
                        st.markdown(message["content"])
                        
                        # Show sources if available
                        if "sources" in message and message["sources"]:
                            with st.expander("📖 View Sources"):
                                for i, source in enumerate(message["sources"][:3], 1):
                                    source_name = source.get("source", "Unknown")
                                    source_type = source.get("type", "")
                                    page = source.get("page")
                                    subtype = source.get("doc_subtype", "")
                                    # Build a human-readable detail line
                                    detail_parts = [source_name]
                                    if source_type == "pdf" and page:
                                        detail_parts.append(f"Page {page}")
                                    elif source_type == "csv" and subtype:
                                        detail_parts.append(subtype.replace("_", " ").title())
                                    st.markdown(f"**Source {i}** — {' · '.join(detail_parts)}")
                                    st.caption(source.get("content", "")[:300] + "...")
        else:
            st.info("👋 Start by typing a question about AMR data, trends, or upload your own data!")
    
    # User input
    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([0.9, 0.1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question about AMR data...",
                placeholder="E.g., What are the latest resistance trends for Ciprofloxacin?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit = st.form_submit_button("📤", use_container_width=True)
        
        if submit and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                try:
                    response, sources = st.session_state.rag_system.chat(user_input)
                except ValueError as e:
                    response = f"❌ Error: {str(e)}"
                    sources = []
                except Exception as e:
                    response = f"❌ Unexpected error: {str(e)}"
                    sources = []
            
            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "sources": sources[:3]  # Store top 3 sources
            })
            
            st.rerun()
    
    # Display usage tips
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        1. **Be Specific:** Instead of "Tell me about resistance", try "What are the resistance patterns for Ampicillin in Lagos?"
        2. **Ask Comparative Questions:** "How does Ciprofloxacin resistance compare to Gentamicin?"
        3. **Request Analysis:** "What factors contribute most to resistance of X antibiotic?"
        4. **Use Your Data:** Upload your CSV to combine insights with our database
        5. **Ask for Recommendations:** "What are the recommended antibiotics based on current resistance patterns?"
        """)



elif selected == 'Data Analysis':
    st.title("Data Analysis " + page_icon)
    st.markdown("""
    Dive into the data and uncover insights about antimicrobial resistance (AMR) in Nigeria! Our data analysis tools allow you to explore trends, identify patterns, and generate visualizations to better understand the AMR landscape in Nigeria. 

    **Key Features:**
    - **Data Exploration:** Browse through our datasets and gain insights into AMR trends, resistance patterns, and more.
    - **Visualization Tools:** Create interactive charts and graphs to visualize AMR data effectively.
    - **Custom Analysis:** Perform custom analyses to answer specific research questions or explore hypotheses.

    Start analyzing the data and contribute to the fight against antimicrobial resistance in Nigeria!
    """)
    
    # Load data for this page
    try:
        df = load_data("cleaned/amr_nigeria_full_data.csv")
    except FileNotFoundError:
        st.error("❌ Data file not found!")
        st.info("Please ensure `cleaned/amr_nigeria_full_data.csv` exists in the project folder.")
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")
    antibiotics = sorted(df['Antibiotic'].unique())
    selected_antibiotic = st.sidebar.selectbox("Select Antibiotic", antibiotics)

    # Filter by species (multi-select)
    species_options = sorted(df[df['Antibiotic']==selected_antibiotic]['Species'].unique())
    selected_species = st.sidebar.multiselect("Select Species (optional)", species_options)

    # Year range
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

    # Apply filters
    mask = (df['Antibiotic'] == selected_antibiotic) & (df['Year'].between(year_range[0], year_range[1]))
    if selected_species:
        mask &= df['Species'].isin(selected_species)
    filtered_df = df[mask]

    # Main content
    st.title("AMR Data Analysis Dashboard")
    st.markdown(f"### Analyzing: **{selected_antibiotic}**")
    st.write(f"Showing {len(filtered_df)} isolates after filters.")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Resistance Patterns", "Demographics", "Clinical Factors", "Genomics"])

    with tab1:
        st.subheader("Resistance Over Time")
        fig, obs, imp = plot_trend(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications for Stakeholders"):
                st.markdown(f"**Policymakers:** {imp['policymakers']}")
                st.markdown(f"**Clinicians:** {imp['clinicians']}")
                st.markdown(f"**Researchers:** {imp['researchers']}")
                st.markdown(f"**Public:** {imp['public']}")
        else:
            st.warning(obs)

        st.subheader("Resistance by Species")
        fig, obs, imp = plot_by_species(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("Resistance by Age Group")
        fig, obs, imp = plot_by_age(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("Resistance by Gender")
        fig, obs, imp = plot_by_gender(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("Resistance by Specialty")
        fig, obs, imp = plot_by_specialty(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("Resistance by Source")
        fig, obs, imp = plot_by_source(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)
    
    with tab2:
        st.subheader("Species Distribution by Gender")
        fig, obs, imp = plot_species_by_gender(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("Species Distribution by Age Group")
        fig, obs, imp = plot_species_by_age(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("Species Distribution Over Years")
        fig, obs, imp = plot_species_over_years(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("Species Distribution by Susceptibility")
        fig, obs, imp = plot_species_by_susceptibility(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("MIC Distribution by Species")
        fig, obs, imp = plot_mic_by_species(filtered_df, selected_antibiotic)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)
    
    with tab3:
        # If species selected, show trend for that species; otherwise prompt
        if selected_species:
            st.subheader("Species-Specific Susceptibility Trend Over Time")
            # For simplicity, we'll allow user to pick a species from the selected ones
            species_choice = st.selectbox("Choose a species to view its trend", selected_species)
            fig, obs, imp = plot_species_trend_by_susceptibility(filtered_df, selected_antibiotic, species_choice)
            if fig:
                st.plotly_chart(fig, width='stretch')
                st.info(obs)
                with st.expander("Implications"):
                    for k, v in imp.items():
                        st.markdown(f"**{k.capitalize()}:** {v}")
            else:
                st.warning(obs)
        else:
            st.info("Please select at least one species in the sidebar to view species-specific trends.")

    with tab4:
        st.subheader("Genotype-Phenotype Correlation")
        # List of genes present in the dataset (you may need to extract)
        # For demo, we'll hardcode common genes; better to derive from data
        available_genes = ['NDM', 'CTX-M', 'TEM', 'IMP', 'OXA']  # adjust
        gene_choice = st.selectbox("Select Resistance Gene", available_genes)
        fig, obs, imp = plot_by_gene(filtered_df, selected_antibiotic, gene_choice)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

        st.subheader("MIC Distribution")
        species_for_mic = st.selectbox("Filter by species (optional)", ['All'] + list(filtered_df['Species'].unique()))
        species_arg = None if species_for_mic == 'All' else species_for_mic
        fig, obs, imp = plot_mic_distribution(filtered_df, selected_antibiotic, species_arg)
        if fig:
            st.plotly_chart(fig, width='stretch')
            st.info(obs)
            with st.expander("Implications"):
                for k, v in imp.items():
                    st.markdown(f"**{k.capitalize()}:** {v}")
        else:
            st.warning(obs)

elif selected == 'Statistical Analysis':
    st.title("Bayesian Hierarchical Model (BHM) Analysis " + page_icon)
    st.markdown("""
    Advanced Bayesian statistical insights into antimicrobial resistance (AMR) in Nigeria.
    These analyses employ hierarchical Bayesian modeling with PyMC and ArviZ to extract posterior distributions
    and posterior predictive trajectories, providing robust uncertainty quantification.

    **Analyses:**
    - **Nigerian Resistance Risk Score (NRRS)**: Isolate-level resistance probability incorporating species, genes, and demographics
    - **Genotype-Phenotype Discordance**: Detection of resistance gene-phenotype mismatches (silent genes, cryptic mechanisms)
    - **Cross-Species HGT Networks**: Posterior probability of horizontal gene transfer across bacterial species
    - **Temporal Evolution**: Bayesian trend estimation for species-specific resistance trajectories
    """)
    
    # Select antibiotic for BHM analysis
    available_antibiotics = get_available_antibiotics()
    if len(available_antibiotics) == 0:
        st.error("No Bayesian Hierarchical Model results available. Please run the BHM analysis first.")
    else:
        selected_bhm_antibiotic = st.selectbox("Select Antibiotic for Statistical Analysis", available_antibiotics)
        
        # Create tabs for each BHM analysis
        tab_nrrs, tab_hgt, tab_temporal = st.tabs(["NRRS & Discordance", "HGT Networks", "Temporal Evolution"])
        
        # =====================================================================
        # TAB 1: NRRS & Genotype-Phenotype Discordance Detection
        # =====================================================================
        with tab_nrrs:
            st.subheader("Nigerian Resistance Risk Score (NRRS) & Genotype-Phenotype Discordance")
            
            # Load NRRS data
            nrrs_data = load_nrrs_analysis(selected_bhm_antibiotic)
            
            if nrrs_data:
                # Display NRRS distribution visualization
                st.markdown("### 1. NRRS Score Distribution")
                nrrs_img_path = f"bhm_results/{selected_bhm_antibiotic}/nrrs_distribution.png"
                if os.path.exists(nrrs_img_path):
                    st.image(nrrs_img_path, width='stretch', caption="Histogram of Nigerian Resistance Risk Scores across all isolates")
                else:
                    st.warning(f"Visualization not found: {nrrs_img_path}")
                
                # NRRS Observation
                obs_nrrs, imp_nrrs = generate_nrrs_observation(selected_bhm_antibiotic, nrrs_data)
                st.markdown(obs_nrrs)
                
                with st.expander("Implications for Stakeholders"):
                    st.markdown(f"**Policymakers:** {imp_nrrs['policymakers']}")
                    st.markdown(f"**Clinicians:** {imp_nrrs['clinicians']}")
                    st.markdown(f"**Researchers:** {imp_nrrs['researchers']}")
                    st.markdown(f"**Public:** {imp_nrrs['public']}")
                
                # Display predictions vs observed
                st.markdown("### 2. Predicted vs. Observed Resistance")
                pred_vs_obs_path = f"bhm_results/{selected_bhm_antibiotic}/predictions_vs_obs.png"
                if os.path.exists(pred_vs_obs_path):
                    st.image(pred_vs_obs_path, width='stretch', caption="Scatter plot of predicted resistance probability vs. observed phenotype, colored by presence of key resistance genes")
                
                # Discordance summary
                st.markdown("### 3. Genotype-Phenotype Discordance by Species")
                discord_by_species = nrrs_data['discord_by_species'].copy()
                st.dataframe(
                    discord_by_species.style.format({
                        'Discordance_Rate_%': '{:.2f}%'
                    }),
                    width='stretch'
                )
                
                # Data-driven critical analysis
                total_discord = discord_by_species['Discordant_Count'].sum()
                total_isolates = discord_by_species['Total_Isolates'].sum()
                discord_rate = (total_discord / total_isolates) * 100 if total_isolates > 0 else 0
                top_discord = discord_by_species.nlargest(2, 'Discordance_Rate_%')
                low_discord = discord_by_species.nsmallest(2, 'Discordance_Rate_%')
                
                # Extract values safely
                top_species = top_discord.iloc[0]['Species'] if len(top_discord) > 0 else 'Unknown'
                top_discord_rate = top_discord.iloc[0]['Discordance_Rate_%'] if len(top_discord) > 0 else 0.0
                low_species = low_discord.iloc[0]['Species'] if len(low_discord) > 0 else 'Unknown'
                low_discord_rate = low_discord.iloc[0]['Discordance_Rate_%'] if len(low_discord) > 0 else 0.0
                
                st.info(
                    f"**Critical Bayesian Analysis: Diagnostic Reliability & Surveillance Bias Detection**\n\n"
                    f"**Key Questions Addressed:**\n\n"
                    f"1. **Can we trust molecular detection for clinical decisions?** "
                    f"Network-wide discordance rate: {discord_rate:.1f}% ({total_discord}/{total_isolates} isolates). "
                    f"This means {discord_rate:.1f}% of cases show gene-phenotype mismatch. "
                    f"**Clinical implication**: Rapid molecular testing alone risks both false negatives (missing resistant phenotypes) and false positives (treating susceptible organisms). "
                    f"In {top_species}, discordance reaches {top_discord_rate:.1f}%—highest risk for misdiagnosis.\n\n"
                    f"2. **Which species show concordant results?** "
                    f"{low_species} has only {low_discord_rate:.1f}% discordance—genotype reliably predicts phenotype here. "
                    f"This species may be suitable for genetic screening as first-line test, but Bayesian posterior probabilities reveal this is data-dependent.\n\n"
                    f"3. **What underlying mechanisms cause discordance?** "
                    f"Silent/cryptic genes (present but non-functional) are more common than cryptic phenotypes (resistant without detected genes). "
                    f"In Nigeria's underdiagnosis context, this means surveillance systems may be systematically **under-estimating resistance prevalence** by relying on gene detection alone.\n\n"
                    f"**Bayesian Insight**: Discordance rates are posterior estimates incorporating uncertainty from small sample sizes per species. "
                    f"Species with few isolates show wider credible intervals—use caution interpreting high rates from limited data. "
                    f"Focus intervention on species with high discordance AND adequate sample size."
                )
            else:
                st.error(f"NRRS data for {selected_bhm_antibiotic} could not be loaded.")
        
        # =====================================================================
        # TAB 2: Cross-Species Resistance Transfer (HGT Networks)
        # =====================================================================
        with tab_hgt:
            st.subheader("Cross-Species Resistance Transfer Signatures (Horizontal Gene Transfer)")
            
            # Load gene sharing data
            gene_data = load_gene_sharing_analysis(selected_bhm_antibiotic)
            
            if gene_data:
                # Display gene sharing probabilities visualization
                st.markdown("### 1. Gene Sharing Across Species")
                gene_sharing_img_path = f"bhm_results/{selected_bhm_antibiotic}/gene_sharing_probabilities.png"
                if os.path.exists(gene_sharing_img_path):
                    st.image(gene_sharing_img_path, width='stretch', caption="Posterior probability that resistance genes are shared across at least 2 bacterial species")
                
                # HGT Observation
                obs_hgt, imp_hgt = generate_hgt_observation(selected_bhm_antibiotic, gene_data)
                st.markdown(obs_hgt)
                
                with st.expander("Implications for Stakeholders"):
                    st.markdown(f"**Policymakers:** {imp_hgt['policymakers']}")
                    st.markdown(f"**Clinicians:** {imp_hgt['clinicians']}")
                    st.markdown(f"**Researchers:** {imp_hgt['researchers']}")
                    st.markdown(f"**Public:** {imp_hgt['public']}")
                
                # Gene-specific species probabilities
                st.markdown("### 2. Per-Gene Species Distribution")
                gene_choice = st.selectbox("Select Gene", gene_data['summary']['Gene'].unique())
                
                if gene_choice in gene_data['per_gene']:
                    gene_probs = gene_data['per_gene'][gene_choice]
                    st.dataframe(
                        gene_probs.style.format({
                            'Median_Probability_Gene_Presence': '{:.4f}'
                        }),
                        width='stretch'
                    )
                    
                    # Display per-gene visualization
                    gene_viz_path = f"bhm_results/{selected_bhm_antibiotic}/gene_{gene_choice}_species_probs.png"
                    if os.path.exists(gene_viz_path):
                        st.image(gene_viz_path, width='stretch', caption=f"Posterior probability of {gene_choice} presence across species")
                    
                    # Data-driven critical analysis of per-gene distribution
                    top_carriers = gene_probs.nlargest(2, 'Median_Probability_Gene_Presence')
                    rare_carriers = gene_probs[gene_probs['Median_Probability_Gene_Presence'] < 0.10]
                    
                    if len(top_carriers) > 0:
                        top_carrier_species = top_carriers.iloc[0]['Species']
                        top_carrier_prob = top_carriers.iloc[0]['Median_Probability_Gene_Presence']
                    else:
                        top_carrier_species = "Unknown"
                        top_carrier_prob = 0.0
                    
                    st.info(
                        f"**Critical Analysis: {gene_choice} Distribution & Horizontal Gene Transfer Signatures**\n\n"
                        f"**Key Questions Addressed:**\n\n"
                        f"1. **Which species are primary reservoirs?** "
                        f"{top_carrier_species} has the highest posterior probability ({top_carrier_prob:.3f}) of carrying {gene_choice}. "
                        f"**Public health implication**: This species is a critical surveillance target. "
                        f"In Nigeria's setting, the {top_carrier_species} population (clinical and environmental) should be prioritized for infection control and surveillance resources.\n\n"
                        f"2. **Is {gene_choice} shared across species (HGT signature)?** "
                        f"Number of species with near-certain presence (P>0.90): {len(gene_probs[gene_probs['Median_Probability_Gene_Presence'] > 0.90])} out of {len(gene_probs)}. "
                        f"Wide sharing indicates active horizontal gene transfer—resistance is mobile and can rapidly spread across the microbiome. "
                        f"Narrow distribution suggests species-specific adaptation or recent acquisition not yet spread.\n\n"
                        f"3. **Which species are emerging threats?** "
                        f"Species with intermediate probability ({len(gene_probs[(gene_probs['Median_Probability_Gene_Presence'] >= 0.30) & (gene_probs['Median_Probability_Gene_Presence'] < 0.70)])}) are on trajectory toward fixation. "
                        f"These warrant predictive surveillance to prevent future clinical dominance.\n\n"
                        f"**Bayesian Insight**: Posterior probabilities reflect evidence from genotyping studies in the bhm_results. "
                        f"Wide credible intervals for a species indicate sparse sequencing data—uncertain estimates. "
                        f"Narrow intervals (certain presence or absence) reflect robust evidence. Use interval width to guide surveillance prioritization."
                    )
            else:
                st.error(f"Gene sharing data for {selected_bhm_antibiotic} could not be loaded.")
        
        # =====================================================================
        # TAB 3: Temporal Evolution of Resistance
        # =====================================================================
        with tab_temporal:
            st.subheader("Temporal Evolution of Antibiotic Resistance")
            st.markdown(
                "Bayesian hierarchical model estimates posterior trends in resistance over time, "
                "accounting for species heterogeneity and study-level variability."
            )
            
            # Load temporal data
            temporal_data = load_temporal_analysis(selected_bhm_antibiotic)
            
            if temporal_data:
                # Display national trend visualization
                st.markdown("### 1. National Resistance Trend")
                national_trend_path = f"bhm_results/{selected_bhm_antibiotic}/{selected_bhm_antibiotic}_national_trend.png"
                if os.path.exists(national_trend_path):
                    st.image(national_trend_path, width='stretch', caption="Posterior national trend in resistance over time with credible intervals")
                
                # Temporal Observation
                obs_temporal, imp_temporal = generate_temporal_observation(selected_bhm_antibiotic, temporal_data)
                st.markdown(obs_temporal)
                
                with st.expander("Implications for Stakeholders"):
                    st.markdown(f"**Policymakers:** {imp_temporal['policymakers']}")
                    st.markdown(f"**Clinicians:** {imp_temporal['clinicians']}")
                    st.markdown(f"**Researchers:** {imp_temporal['researchers']}")
                    st.markdown(f"**Public:** {imp_temporal['public']}")
                
                # Species-specific slopes
                st.markdown("### 2. Species-Specific Resistance Trends")
                species_slopes = temporal_data['species_slopes'].copy()
                species_slopes = species_slopes.sort_values('Median_Slope_Resistance_Change_per_Year', ascending=False)
                
                st.dataframe(
                    species_slopes.style.format({
                        'Median_Slope_Resistance_Change_per_Year': '{:.4f}',
                        'Probability_Slope_Positive': '{:.4f}'
                    }).background_gradient(
                        subset=['Probability_Slope_Positive'],
                        cmap='RdYlGn_r',
                        vmin=0, vmax=1
                    ),
                    width='stretch'
                )
                
                
                # Data-driven critical analysis of trends
                accelerating = species_slopes[species_slopes['Probability_Slope_Positive'] > 0.95]
                decelerating = species_slopes[species_slopes['Probability_Slope_Positive'] < 0.05]
                uncertain = species_slopes[(species_slopes['Probability_Slope_Positive'] >= 0.40) & (species_slopes['Probability_Slope_Positive'] <= 0.60)]
                
                # Extract values safely
                top_accelerator = accelerating.iloc[0]['Species'] if len(accelerating) > 0 else 'None'
                top_accel_slope = accelerating.iloc[0]['Median_Slope_Resistance_Change_per_Year'] if len(accelerating) > 0 else 0.0
                
                st.info(
                    f"**Critical Bayesian Analysis: Species-Specific Resistance Dynamics**\n\n"
                    f"**Color Legend (Probability_Slope_Positive Column):**\n"
                    f"- 🔴 **Red (P ≈ 0)**: High-confidence decreasing trend (P < 0.05). Resistance is declining.\n"
                    f"- 🟡 **Yellow (P ≈ 0.5)**: Uncertain trend. Insufficient evidence to declare increase or decrease.\n"
                    f"- 🟢 **Green (P ≈ 1)**: High-confidence increasing trend (P > 0.95). Resistance is accelerating.\n\n"
                    f"**Key Questions Addressed:**\n\n"
                    f"1. **Which species pose accelerating clinical threats?** "
                    f"**High-confidence increasing trends (P(slope>0) > 0.95)**: {len(accelerating)} species. "
                    f"Top accelerator: {top_accelerator} ({top_accel_slope:.4f} log-odds/year). "
                    f"These represent urgent public health crises—resistance is evolving rapidly and may outpace stewardship interventions in Nigeria.\n\n"
                    f"2. **How certain are these trends?** "
                    f"**Uncertain trajectories (P ≈ 0.50)**: {len(uncertain)} species have conflicting signals. "
                    f"This indicates either true stability (resistance plateau) or sparse/noisy surveillance data. "
                    f"In Nigeria's context, ambiguous trends should trigger enhanced surveillance rather than complacency. "
                    f"Conversely, **high-confidence decreasing trends (P < 0.05)**: {len(decelerating)} species merit celebration but verification; declining resistance may reflect improved infection control, reduced antibiotic exposure, or surveillance artifact.\n\n"
                    f"3. **Do species differ in resistance architecture?** "
                    f"Slope range: {species_slopes['Median_Slope_Resistance_Change_per_Year'].min():.4f} to {species_slopes['Median_Slope_Resistance_Change_per_Year'].max():.4f} log-odds/year. "
                    f"This wide variance reveals species respond differently to selective pressures. "
                    f"High slopes (clonal emergence or rapid gene spread) demand immediate molecular investigation. "
                    f"Shallow slopes suggest baseline resistance with slow evolution (endemic stability).\n\n"
                    f"**Bayesian Insight:** Probability_Slope_Positive quantifies posterior belief in increasing resistance, borrowing strength across all species via hierarchical structure. "
                    f"P=0.95 means 95% of posterior MCMC samples show positive slope—overwhelming evidence. "
                    f"P≈0.50 means equal credence to increase/decrease; without additional data, assume neutral until proven otherwise.\n\n"
                    f"**Nigeria Epidemiology Context:** Surveillance bias is critical. Negative slopes may reflect facility closures, reduced testing, or demographic shifts rather than true resistance decline. "
                    f"Positive slopes in over-tested populations may be artifactual. Cross-validate with independent data sources (WHO, cross-border labs) before policy response."
                )
                
                # Species trajectories visualization
                st.markdown("### 3. Species Trajectories (Top Increasing/Decreasing)")
                species_traj_path = f"bhm_results/{selected_bhm_antibiotic}/{selected_bhm_antibiotic}_species_trajectories_selected.png"
                if os.path.exists(species_traj_path):
                    st.image(species_traj_path, width='stretch', caption="Top accelerating and decelerating species with posterior credible intervals")
                
                # Model summary
                st.markdown("### 4. Bayesian Model Summary (Posterior Estimates)")
                model_summary = temporal_data['model_summary'].copy()
                st.dataframe(
                    model_summary.style.format('{:.4f}'),
                    width='stretch'
                )
                
                # Data-driven Bayesian diagnostics
                converged = (model_summary['r_hat'] < 1.01).sum()
                problematic = (model_summary['r_hat'] > 1.05).sum()
                national_slope = model_summary.loc['national_slope', model_summary.columns[0]] if 'national_slope' in model_summary.index else None
                national_slope_str = f"{national_slope:.4f}" if national_slope is not None else "N/A"
                
                st.info(
                    f"**Critical Bayesian Model Diagnostics & Interpretation**\n\n"
                    f"**Convergence Assessment (r-hat):** {converged}/{len(model_summary)} parameters converged (<1.01)—posterior samples are reliable. "
                    f"{problematic} parameters show r-hat > 1.05 (potential non-stationarity). "
                    f"**Action if problematic**: Re-run hierarchical model with more MCMC iterations or investigate data anomalies (e.g., single dominant study or missing temporal data for a species).\n\n"
                    f"**Variance Decomposition—What drives resistance?** "
                    f"(σ_species)² ÷ total variance reveals attributable fractions. "
                    f"**High σ_species**: Species biology dominates—intrinsic susceptibility or acquired resistance traits differ fundamentally. Search for genetic markers. "
                    f"**High σ_study**: Surveillance protocol/geographic variation dominates—inconsistent lab practices or regional transmission clusters. Harmonize protocols. "
                    f"In multi-lab Nigerian networks, decomposing variance is essential to avoid misinterpreting geographic patterns as biological differences.\n\n"
                    f"**National Slope (Population-Level Trend):** Posterior median ~{national_slope_str} log-odds/year (if available). "
                    f"This averaged slope masks species heterogeneity; refer to species-specific slopes for clinical action. "
                    f"Credible interval width tells you if this estimate is definitive (narrow: robust evidence) or tentative (wide: sparse data).\n\n"
                    f"**Nigeria Surveillance Implications:** Hierarchical structure prevents urban-center bias—rural/under-resourced facilities' data is weighted proportionally. "
                    f"However, if one region reports consistently, its influence may dominate. Wide credible intervals for national estimates often reflect uneven geographic coverage. "
                    f"Propose: Expand surveillance to underrepresented regions (northern states, private labs) to narrow intervals and improve national picture clarity."
                )
            else:
                st.error(f"Temporal data for {selected_bhm_antibiotic} could not be loaded.")
        

elif selected == 'Train Model':
    train_model()

elif selected == 'Make Prediction':
    make_prediction()

elif selected == 'About':
    st.title("About " + page_icon)
    st.markdown("""
    The Nigeria AMR Hub is a collaborative initiative aimed at addressing the challenges of antimicrobial resistance (AMR) in Nigeria. Our platform provides a centralized resource for researchers, healthcare professionals, policymakers, and the general public to access data, insights, and tools related to AMR in Nigeria. 

    **Our Mission:**
    To combat antimicrobial resistance in Nigeria by providing comprehensive data, analysis tools, and predictive models to inform decision-making and drive evidence-based interventions.

    **Our Team:**
    We are a multidisciplinary team of researchers, data scientists, healthcare professionals, and policymakers dedicated to addressing the challenges of AMR in Nigeria. Our team collaborates with local and international partners to ensure that our platform is up-to-date and relevant to the needs of our users.

    **Contact Us:**
    If you have any questions, suggestions, or would like to collaborate with us, please feel free to reach out! You can contact us at [email address] or follow us on social media for updates and news about our work.

    Join us in our efforts to combat antimicrobial resistance and improve public health outcomes in Nigeria!
    """)

