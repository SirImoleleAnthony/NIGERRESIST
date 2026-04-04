# Make Prediction with XGBoost & Conformal Prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from io import BytesIO
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

from dowhy import CausalModel
from conformal_utils import (
    load_antibiotic_data,
    prepare_data_for_modeling,
    train_conformal_model,
    create_conformal_report,
    CREPES_AVAILABLE
)


def make_prediction():

    st.title("🔮 Make Individual Resistance Predictions")
    
    st.markdown("""
    ### Predict Antimicrobial Resistance Status
    
    This interface allows you to:
    1. **Select antibiotic** and **species** from cleaned data
    2. **Choose specific parameters** (study type, source, phenotype, etc.) for your case
    3. **Get XGBoost prediction** for resistance status
    4. **Understand prediction confidence** via Conformal Prediction intervals
    5. **(Optional)** Check causal factors influencing the prediction
    """)
    
    # Get available antibiotics from cleaned datasets
    cleaned_dir = Path("cleaned")
    antibiotic_files = sorted([
        f.name.replace("_Nigerian_subset_cleaned.csv", "") 
        for f in cleaned_dir.glob("*_Nigerian_subset_cleaned.csv")
        if "_I_" not in f.name  # Exclude target variable files
    ])
    
    if not antibiotic_files:
        st.error("❌ No cleaned antibiotic datasets found in 'cleaned/' folder.")
        return
    
    # Step 1: Antibiotic Selection
    st.subheader("1️⃣ Select Antibiotic & Species")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_antibiotic = st.selectbox(
            "Choose antibiotic:",
            antibiotic_files,
            help="Select antibiotic to predict resistance for"
        )
    
    # Load data
    df_loaded = load_antibiotic_data(selected_antibiotic, use_cleaned=True)
    if df_loaded is None:
        st.error(f"❌ Could not load data for {selected_antibiotic}")
        return
    
    with col2:
        species_list = sorted(df_loaded['Species'].unique().tolist())
        if not species_list:
            st.error("No species found in dataset")
            return
        
        selected_species = st.selectbox(
            "Choose species:",
            species_list,
            help="Select bacterial species to predict for"
        )
    
    st.success(f"✅ Loaded {len(df_loaded)} records for {selected_antibiotic}")
    
    # Prepare training data
    data_prep = prepare_data_for_modeling(df_loaded, selected_antibiotic, selected_species)
    X_train = data_prep['X']
    y_train = data_prep['y']
    features = data_prep['features']
    le_outcome = data_prep['le_outcome']
    encoding_dict = data_prep['encoding_dict']
    
    # Remove Country and Species from features (Nigeria-only and user already selected species respectively)
    features = [f for f in features if f not in ['Country', 'Species']]
    if 'Country' in X_train.columns:
        X_train = X_train.drop('Country', axis=1)
    if 'Species' in X_train.columns:
        X_train = X_train.drop('Species', axis=1)
    
    if len(X_train) < 15:
        st.warning(f"⚠️ Only {len(X_train)} samples for {selected_species}. Need ≥ 15 to train model.")
        return
    
    unique_classes = len(np.unique(y_train))
    if unique_classes < 2:
        st.warning(f"⚠️ Only 1 resistance class for {selected_species} → {selected_antibiotic}. Cannot make meaningful predictions.")
        return
    
    st.info(f"✓ {len(X_train)} samples | {len(features)} features | {unique_classes} resistance classes")
    
    # Initialize session state for predictions
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'last_prediction_params' not in st.session_state:
        st.session_state.last_prediction_params = None
    
    # Step 2: Conformal Prediction Settings
    st.subheader("2️⃣ Prediction Confidence Settings")
    
    alpha = st.slider(
        "Confidence Level (1 - α)",
        min_value=0.80,
        max_value=0.99,
        value=0.90,
        step=0.01,
        help="Higher = more confident predictions. At 0.90, predictions are valid 90% of the time."
    )
    
    # Step 3: User Input for Single Sample Prediction
    st.subheader("3️⃣ Select Parameters for Prediction")
    
    # Create columns for better UI layout
    col1, col2, col3 = st.columns(3)
    
    feature_inputs = {}
    col_idx = 0
    columns = [col1, col2, col3]
    
    for feature in features:
        col = columns[col_idx % 3]
        
        with col:
            if feature in X_train.columns:
                # Get unique values for this feature
                unique_vals = df_loaded[feature].unique()
                unique_vals = [str(v) for v in unique_vals if pd.notna(v)]
                
                if feature in encoding_dict:
                    # Categorical feature - show decoded values
                    le = encoding_dict[feature]
                    
                    if len(unique_vals) > 20:
                        feature_inputs[feature] = st.selectbox(
                            f"**{feature}**",
                            sorted(unique_vals)[:50],
                            key=f"input_{feature}"
                        )
                    else:
                        feature_inputs[feature] = st.selectbox(
                            f"**{feature}**",
                            sorted(unique_vals),
                            key=f"input_{feature}"
                        )
                else:
                    # Numeric feature
                    try:
                        min_val = float(df_loaded[feature].min())
                        max_val = float(df_loaded[feature].max())
                        feature_inputs[feature] = st.slider(
                            f"**{feature}**",
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int((min_val + max_val) / 2),
                            key=f"input_{feature}"
                        )
                    except:
                        feature_inputs[feature] = st.number_input(
                            f"**{feature}**",
                            value=1.0,
                            key=f"input_{feature}"
                        )
        
        col_idx += 1
    
    # Optional: Causal Analysis
    st.subheader("4️⃣ Additional Analysis (Optional)")
    
    show_causal = st.checkbox(
        "🔗 Enable Causal Effect Analysis",
        value=False,
        help="Understand which factors causally influence the prediction"
    )
    
    # Make Prediction Button
    st.markdown("---")
    
    if st.button("🚀 Make Prediction", width='stretch', type="primary"):
        # Store parameters to session state
        st.session_state.prediction_params = {
            'antibiotic': selected_antibiotic,
            'species': selected_species,
            'alpha': alpha,
            'features': features,
            'feature_inputs': feature_inputs,
            'show_causal': show_causal
        }
        
        with st.spinner(f"Training XGBoost model and making prediction for {selected_species} → {selected_antibiotic}..."):
            
            # Initialize optional result variables
            conformal_clf = None
            p_values_input = None
            cp_results = None
            max_pvalue = None
            confidence_level = None
            
            # Train XGBoost model
            clf = XGBClassifier(
                objective='multi:softmax',
                num_class=unique_classes,
                eval_metric='mlogloss',
                random_state=42,
                verbosity=0,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100
            )
            
            # Split data for CP training
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Train base model
            clf.fit(X_train_split, y_train_split)
            
            # Evaluate on test set (for metrics only)
            y_pred_test = clf.predict(X_test_split)
            accuracy = accuracy_score(y_test_split, y_pred_test)
            precision = precision_score(y_test_split, y_pred_test, average='weighted', zero_division=0)
            recall = recall_score(y_test_split, y_pred_test, average='weighted', zero_division=0)
            f1 = f1_score(y_test_split, y_pred_test, average='weighted', zero_division=0)
            
            # Prepare user input for prediction
            X_input = pd.DataFrame([feature_inputs])
            
            # Encode user input using the same encoders
            for col in X_input.columns:
                if col in encoding_dict:
                    le = encoding_dict[col]
                    try:
                        X_input[col] = le.transform(X_input[col].astype(str))
                    except:
                        # If value not in encoder, use 0
                        X_input[col] = 0
                elif col == 'Year':
                    X_input[col] = float(X_input[col])
            
            # Make prediction with base model
            pred_class_idx = clf.predict(X_input)[0]
            pred_proba = clf.predict_proba(X_input)[0]
            
            # Get prediction label
            if le_outcome:
                pred_label = le_outcome.classes_[pred_class_idx]
            else:
                pred_label = pred_class_idx
            
            # Create prediction probability dataframe for display
            prob_df = pd.DataFrame({
                'Resistance Status': le_outcome.classes_ if le_outcome else [f'Class {i}' for i in range(len(pred_proba))],
                'Probability': pred_proba
            }).sort_values('Probability', ascending=False)
            
            # Conformal Prediction for Uncertainty Quantification
            if CREPES_AVAILABLE:
                # Train conformal model
                conformal_clf, y_pred_conf, predicted_sets, p_values = train_conformal_model(
                    X_train_split, y_train_split, X_test_split, y_test_split, clf, alpha=1-alpha
                )
                
                if conformal_clf is not None:
                    # Generate CP prediction for user input
                    p_values_input = conformal_clf.predict_p(X_input)
                    
                    # Create CP results dataframe
                    cp_results = pd.DataFrame({
                        'Resistance Status': le_outcome.classes_ if le_outcome else [f'Class {i}' for i in range(len(p_values_input[0]))],
                        'Confidence Score (p-value)': p_values_input[0]
                    }).sort_values('Confidence Score (p-value)', ascending=False)
                    
                    # Determine confidence level
                    max_pvalue = cp_results['Confidence Score (p-value)'].max()
                    if max_pvalue > 0.7:
                        confidence_level = "High"
                    elif max_pvalue > 0.5:
                        confidence_level = "Medium"
                    else:
                        confidence_level = "Low"
            
            # Compute feature importance
            fi_df = pd.DataFrame({
                'Feature': features,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            # Store results in session state
            st.session_state.prediction_results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pred_class_idx': pred_class_idx,
                'pred_proba': pred_proba,
                'pred_label': pred_label,
                'prob_df': prob_df,
                'conformal_model': conformal_clf if CREPES_AVAILABLE else None,
                'p_values_input': p_values_input if CREPES_AVAILABLE and conformal_clf is not None else None,
                'cp_results': cp_results if CREPES_AVAILABLE and conformal_clf is not None else None,
                'max_pvalue': max_pvalue if CREPES_AVAILABLE and conformal_clf is not None else None,
                'confidence_level': confidence_level if CREPES_AVAILABLE and conformal_clf is not None else None,
                'fi_df': fi_df,
                'clf': clf,
                'X_test_split': X_test_split,
                'y_test_split': y_test_split
            }
            
            st.success("✅ Prediction complete! Results displayed below.")
    
    # Display prediction results from session state (persists across reruns)
    if st.session_state.prediction_results is not None:
        results = st.session_state.prediction_results
        
        st.subheader("📊 Model Performance on Test Set")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{results['precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{results['recall']*100:.2f}%")
        with col4:
            st.metric("F1 Score", f"{results['f1']*100:.2f}%")
        
        # Display Main Prediction
        st.subheader("🎯 Prediction Result")
        
        # Color code by resistance status
        if results['pred_label'] == 'Resistant':
            st.error(f"### ⚠️ **RESISTANT**")
            st.markdown(f"""
            The bacterial species **{st.session_state.prediction_params['species']}** is predicted to be **RESISTANT** to **{st.session_state.prediction_params['antibiotic']}**.
            
            This means the antibiotic would likely **NOT be effective** against this organism.
            """)
        elif results['pred_label'] == 'Intermediate':
            st.warning(f"### ⚡ **INTERMEDIATE**")
            st.markdown(f"""
            The bacterial species **{st.session_state.prediction_params['species']}** shows **INTERMEDIATE RESISTANCE** to **{st.session_state.prediction_params['antibiotic']}**.
            
            This means the antibiotic may have **LIMITED effectiveness** depending on dosage and infection type.
            """)
        else:  # Susceptible
            st.success(f"### ✅ **SUSCEPTIBLE**")
            st.markdown(f"""
            The bacterial species **{st.session_state.prediction_params['species']}** is predicted to be **SUSCEPTIBLE** to **{st.session_state.prediction_params['antibiotic']}**.
            
            This means the antibiotic would likely be **EFFECTIVE** against this organism.
            """)
        
        # Show prediction probabilities
        st.markdown("#### Prediction Confidence Breakdown")
        fig_prob = px.bar(
            results['prob_df'],
            x='Probability',
            y='Resistance Status',
            orientation='h',
            color='Resistance Status',
            title=f"Prediction Probabilities for {st.session_state.prediction_params['species']} → {st.session_state.prediction_params['antibiotic']}",
            color_discrete_map={
                'Resistant': '#EF553B',
                'Intermediate': '#FFA15A',
                'Susceptible': '#00CC96'
            }
        )
        st.plotly_chart(fig_prob, width='stretch', key="prediction_prob_chart")
        
        # Conformal Prediction for Uncertainty Quantification
        if results['conformal_model'] is not None:
            st.subheader("🎛️ Conformal Prediction - Uncertainty Intervals")
            
            alpha = st.session_state.prediction_params['alpha']
            st.markdown(f"""
            **Conformal Prediction Confidence (α = {1-alpha}):**
            
            The model is {(1-alpha)*100:.0f}% confident in predictions that fall within these intervals.
            This means at most {alpha*100:.0f}% of test cases will have incorrect predictions.
            """)
            
            # Show p-values for each class
            st.dataframe(results['cp_results'], width='stretch')
            
            # Interpretation
            st.info(f"""
            **Confidence Assessment:** {results['confidence_level']}
            
            - High Confidence (p > 0.7): Model prediction is highly reliable
            - Medium Confidence (0.5 < p ≤ 0.7): Model has moderate uncertainty
            - Low Confidence (p ≤ 0.5): Model is uncertain about prediction
            """)
        
        # Feature Importance
        st.subheader("📈 Feature Importance for This Prediction")
        
        fig_fi = px.bar(
            results['fi_df'],
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top Features Influencing Resistance Prediction',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_fi, width='stretch', key="feature_importance_chart")
        
        st.markdown("""
        **Interpretation:** These features have the strongest influence on predicting 
        resistance status. They represent the most important factors for this species-antibiotic combination.
        """)
        
        # Optional: Causal Analysis (Now with persistent state)
        if st.session_state.prediction_params['show_causal']:
            st.subheader("🔗 Causal Effect Analysis (Optional)")
            
            st.markdown("Analyzing causal relationships between factors and resistance...")
            
            # Load data again for causal analysis
            df_loaded = load_antibiotic_data(st.session_state.prediction_params['antibiotic'], use_cleaned=True)
            
            try:
                _perform_causal_analysis(
                    df_loaded,
                    st.session_state.prediction_params['antibiotic'],
                    st.session_state.prediction_params['species'],
                    st.session_state.prediction_params['features'],
                    None  # Will recreate le_outcome inside
                )
            except Exception as e:
                st.warning(f"Causal analysis encountered an error: {str(e)}")
                st.info("This is optional analysis - your main prediction is still valid.")
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **DISCLAIMER:**
        
        These predictions are generated for **research and educational purposes only**. 
        They should NOT be used for clinical decision-making without validation by domain experts 
        and proper laboratory testing. Always consult with antimicrobial stewardship specialists 
        and follow established clinical guidelines.
        """)


def _perform_causal_analysis(df, antibiotic, species, features, le_outcome):

    outcome_col = f'{antibiotic}_I'
    
    # Filter to species
    df_species = df[df['Species'] == species].copy()
    
    if df_species.empty:
        st.warning(f"No data for {species}")
        return
    
    # Initialize session state for causal analysis
    if 'causal_analysis_run' not in st.session_state:
        st.session_state.causal_analysis_run = False
    if 'causal_results' not in st.session_state:
        st.session_state.causal_results = None
    
    # Select factors for causal analysis
    st.markdown("#### Select Factors for Causal Analysis")
    
    # Exclude Country and Species from causal analysis (data already stratified by species)
    available_factors = [f for f in features if f not in ['Country', 'Species']]
    default_factors = [f for f in ['Phenotype', 'Source', 'Speciality'] if f in available_factors]
    
    selected_factors = st.multiselect(
        "Choose factors to analyze causally:",
        options=available_factors,
        default=default_factors,
        help="Causal analysis will estimate these factors' effect on resistance"
    )
    
    if not selected_factors:
        st.info("Select at least one factor for causal analysis")
        return
    
    if st.button("🔍 Calculate Causal Effects"):
        st.session_state.causal_analysis_run = True
    
    # Run causal analysis if button was clicked and persist results
    if st.session_state.causal_analysis_run:
        
        try:
            with st.spinner("Computing causal effects..."):
                
                # Prepare data
                df_causal = df_species.copy()
                
                # Encode categorical columns
                categorical_cols = [col for col in selected_factors if df_causal[col].dtype == 'object']
                for col in categorical_cols:
                    le = LabelEncoder()
                    df_causal[col] = le.fit_transform(df_causal[col].astype(str))
                
                # Encode outcome
                if df_causal[outcome_col].dtype == 'object':
                    le_outcome_causal = LabelEncoder()
                    df_causal['outcome_encoded'] = le_outcome_causal.fit_transform(df_causal[outcome_col])
                else:
                    df_causal['outcome_encoded'] = df_causal[outcome_col]
                
                # Prepare columns for causal model
                cols_for_model = selected_factors + ['outcome_encoded']
                df_causal_model = df_causal[cols_for_model].dropna()
                
                if len(df_causal_model) < 10:
                    st.warning("Not enough data for reliable causal analysis")
                    st.session_state.causal_analysis_run = False
                    return
                
                causal_effects_list = []
                
                for treatment in selected_factors:
                    try:
                        # Create causal graph
                        graph = nx.DiGraph()
                        confounders = [c for c in selected_factors if c != treatment]
                        
                        graph.add_edge(treatment, 'outcome_encoded')
                        for confounder in confounders:
                            graph.add_edge(confounder, 'outcome_encoded')
                            graph.add_edge(confounder, treatment)
                        
                        # Estimate causal effect
                        model = CausalModel(
                            data=df_causal_model,
                            treatment=[treatment],
                            outcome='outcome_encoded',
                            graph=graph
                        )
                        
                        identified_estimand = model.identify_effect()
                        causal_estimate = model.estimate_effect(
                            identified_estimand,
                            method_name="backdoor.linear_regression",
                            target_units="ate"
                        )
                        
                        causal_effects_list.append({
                            'Factor': treatment,
                            'Causal Effect': float(causal_estimate.value)
                        })
                        
                    except Exception as e:
                        st.warning(f"Could not estimate effect for {treatment}: {str(e)[:100]}")
                
                st.session_state.causal_results = causal_effects_list
        
        except Exception as e:
            st.error(f"Causal analysis error: {str(e)}")
            st.session_state.causal_analysis_run = False
    
    # Display causal results if available
    if st.session_state.causal_results:
        causal_effects = st.session_state.causal_results
        if causal_effects:
            df_effects = pd.DataFrame(causal_effects).sort_values('Causal Effect', ascending=False)
            
            st.markdown("#### Causal Effects Results")
            st.dataframe(df_effects, width='stretch')
            
            # Visualize
            fig_causal = px.bar(
                df_effects,
                x='Causal Effect',
                y='Factor',
                orientation='h',
                title=f'Causal Effects on {antibiotic} Resistance in {species}',
                color='Causal Effect',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_causal, width='stretch', key="causal_effects_chart")
            
            st.info("""
            **Interpretation:**
            - Positive effect: Factor increases resistance risk
            - Negative effect: Factor decreases resistance risk
            - Larger magnitude: Stronger causal influence
            """)
        else:
            st.warning("Could not compute causal effects for selected factors")
