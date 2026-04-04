# AMR Model Training with Conformal Prediction & Explainability

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import conformal prediction utilities
from conformal_utils import (
    load_antibiotic_data, 
    prepare_data_for_modeling,
    train_conformal_model,
    create_conformal_report,
    CREPES_AVAILABLE
)


def train_model(df=None, anti=None):

    st.title("🔬 Train AMR Prediction Model with Conformal Prediction")
    
    st.markdown("""
    ### Advanced Predictive Modeling with Uncertainty & Explainability
    
    This interface allows you to:
    1. **Select antibiotic** and **species** from cleaned data
    2. **Train a classifier** to predict resistance phenotypes
    3. **View conformal prediction confidence intervals** - understand prediction reliability
    4. **Analyze feature contributions** via SHAP - see which factors drove each prediction
    5. **Download trained models** for external analysis
    """)
    
    # Get available antibiotics from cleaned datasets
    from pathlib import Path
    cleaned_dir = Path("cleaned")
    antibiotic_files = sorted([
        f.name.replace("_Nigerian_subset_cleaned.csv", "") 
        for f in cleaned_dir.glob("*_Nigerian_subset_cleaned.csv")
    ])
    
    if not antibiotic_files:
        st.error("❌ No cleaned antibiotic datasets found in 'cleaned/' folder.")
        return
    
    # Step 1: Antibiotic Selection
    st.subheader("1️⃣ Select Antibiotic")
    
    if anti and anti in antibiotic_files:
        selected_antibiotic = anti
        st.info(f"Using antibiotic: **{selected_antibiotic}**")
    else:
        selected_antibiotic = st.selectbox(
            "Choose antibiotic for model training:",
            antibiotic_files,
            index=0,
            help="Selects antibiotic-specific cleaned dataset"
        )
    
    # Load data
    df_loaded = load_antibiotic_data(selected_antibiotic, use_cleaned=True)
    if df_loaded is None:
        st.error(f"❌ Could not load data for {selected_antibiotic}")
        return
    
    st.success(f"✅ Loaded {len(df_loaded)} records for {selected_antibiotic}")
    
    # Step 2: Species Selection
    st.subheader("2️⃣ Select Species")
    species_list = sorted(df_loaded['Species'].unique().tolist())
    
    if len(species_list) == 0:
        st.error(f"No species found in {selected_antibiotic} data")
        return
    
    selected_species = st.selectbox(
        "Choose species for model training:",
        species_list,
        help="Filter to specific bacterial species"
    )
    
    # Prepare data
    data_prep = prepare_data_for_modeling(df_loaded, selected_antibiotic, selected_species)
    X = data_prep['X']
    y = data_prep['y']
    features = data_prep['features']
    le_outcome = data_prep['le_outcome']
    
    if len(X) < 20:
        st.warning(f"⚠️ Only {len(X)} samples for {selected_species}. Model may not train properly (need ≥ 20).")
        return
    
    # Check class distribution
    unique_classes = len(np.unique(y))
    if unique_classes < 2:
        st.warning(f"⚠️ Only 1 resistance class for {selected_species} → {selected_antibiotic}. Cannot train classifier (need ≥ 2 classes).")
        return
    
    st.info(f"✓ {len(X)} samples | {len(features)} features | {unique_classes} resistance classes")
    
    # Step 3: Model Selection
    st.subheader("3️⃣ Select Classification Model")
    
    model_options = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(objective='multi:softmax', num_class=unique_classes, 
                                eval_metric='mlogloss', random_state=42, verbosity=0),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    
    model_name = st.selectbox(
        "Choose classification algorithm:",
        list(model_options.keys()),
        index=0,
        help="Different algorithms have different strengths for AMR prediction"
    )
    
    # Conformal prediction settings
    st.subheader("4️⃣ Conformal Prediction Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        use_conformal = st.checkbox(
            "✓ Enable Conformal Prediction",
            value=CREPES_AVAILABLE,
            disabled=not CREPES_AVAILABLE,
            help="Get prediction confidence intervals" if CREPES_AVAILABLE else "⚠️ CREPES not installed"
        )
    
    with col2:
        alpha = st.slider(
            "Confidence Level (1 - α)",
            min_value=0.80,
            max_value=0.99,
            value=0.90,
            step=0.01,
            help="Higher = more confident predictions, but larger prediction sets"
        )
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05
        )
    
    if not CREPES_AVAILABLE:
        st.warning("⚠️ **CREPES not installed.** Install with: `pip install crepes`")
    
    # Initialize session state for training results
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'training_params' not in st.session_state:
        st.session_state.training_params = None
    
    # Training
    if st.button(f"🚀 Train {model_name} for {selected_species} → {selected_antibiotic}", 
                 width='stretch',
                 type="primary"):
        
        # Store parameters
        st.session_state.training_params = {
            'antibiotic': selected_antibiotic,
            'species': selected_species,
            'model_name': model_name,
            'use_conformal': use_conformal,
            'alpha': alpha
        }
        
        with st.spinner(f"Training {model_name} for {selected_species} resistance to {selected_antibiotic}..."):
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train base model
            clf = model_options[model_name]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Compute metrics (no display)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            class_names = le_outcome.classes_ if le_outcome else np.unique(y)
            
            # Compute feature importance (no display)
            fi_df = None
            if hasattr(clf, 'feature_importances_'):
                fi_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': clf.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
            
            # Conformal Prediction
            conformal_clf = None
            predicted_sets = None
            p_values = None
            conf_report = None
            
            if use_conformal and CREPES_AVAILABLE:
                conformal_clf, y_pred_conf, predicted_sets, p_values = train_conformal_model(
                    X_train, y_train, X_test, y_test, clf, alpha=1-alpha
                )
                
                if conformal_clf is not None:
                    conf_report = create_conformal_report(y_pred, predicted_sets, p_values, y_test, le_outcome)
            
            # Store all results in session state
            st.session_state.training_results = {
                'clf': clf,
                'conformal_clf': conformal_clf,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cm': cm,
                'class_names': class_names,
                'fi_df': fi_df,
                'conf_report': conf_report,
                'features': features,
                'y_test': y_test,
                'y_pred': y_pred,
                'unique_classes': unique_classes
            }
            
            st.success("✅ Training complete!")
    
    # Display training results from session state (persists across reruns)
    if st.session_state.training_results is not None:
        results = st.session_state.training_results
        params = st.session_state.training_params
        
        # Performance Metrics
        st.subheader("📊 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{results['precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{results['recall']*100:.2f}%")
        with col4:
            st.metric("F1 Score", f"{results['f1']*100:.2f}%")
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        fig_cm = px.imshow(
            results['cm'],
            labels=dict(x="Predicted", y="Actual"),
            x=results['class_names'],
            y=results['class_names'],
            color_continuous_scale='Blues',
            text_auto=True,
            title=f"Confusion Matrix: {params['species']} → {params['antibiotic']}"
        )
        st.plotly_chart(fig_cm, width='stretch', key="train_confusion_matrix")
        
        # Feature Importance (if available)
        if results['fi_df'] is not None:
            st.markdown("### Feature Importance")
            
            fig_fi = px.bar(
                results['fi_df'],
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top 10 Features Predicting {params["species"]} Resistance',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_fi, width='stretch', key="train_feature_importance")
            
            st.markdown("""
            **Interpretation**: Features with higher importance scores have greater impact on model predictions.
            These are potential key drivers of resistance in this species.
            """)
        
        # Conformal Prediction
        if params['use_conformal'] and CREPES_AVAILABLE and results['conf_report'] is not None:
            st.markdown("### 🎯 Conformal Prediction Results")
            
            conf_report = results['conf_report']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_set_size = conf_report['Num_Classes_in_Set'].mean()
                st.metric("Avg Prediction Set Size", f"{avg_set_size:.2f}")
            with col2:
                empty_sets = conf_report['Empty_Set'].sum()
                st.metric("Empty Predictions", f"{empty_sets} ({empty_sets/len(conf_report)*100:.1f}%)")
            with col3:
                single_pred = conf_report['Single_Prediction'].sum()
                st.metric("Certain Predictions", f"{single_pred} ({single_pred/len(conf_report)*100:.1f}%)")
            
            # Show sample predictions with confidence
            st.markdown("#### Sample Predictions with Confidence Intervals")
            
            sample_report = conf_report.head(10).copy()
            
            display_cols = [col for col in sample_report.columns if col in 
                           ['Predicted_Class', 'Actual_Class', 'Max_Confidence', 'Confidence_Level', 'Prediction_Type', 'Correct']]
            st.dataframe(sample_report[display_cols], width='stretch')
            
            st.markdown("""
            ### 📊 Conformal Prediction Interpretation Guide:
            
            **Confidence Levels:**
            - **High Confidence** (>0.7): Model prediction is reliable
            - **Medium Confidence** (0.5-0.7): Model has some uncertainty
            - **Low Confidence** (<0.5): Model is uncertain about prediction
            
            **Prediction Types:**
            - **Certain**: Only one resistance class meets the confidence threshold (definitive prediction)
            - **Uncertain**: Multiple resistance classes meet threshold (model is unsure between multiple classes)
            - **Empty**: No classes meet the confidence threshold (model cannot make a confident decision at this threshold)
            
            **Model Performance Summary:**
            """)
            
            # Calculate and display performance metrics
            conf_accuracy = conf_report['Correct'].mean()
            high_conf = (conf_report['Confidence_Level'] == 'High').sum()
            certain = (conf_report['Prediction_Type'] == 'Certain').sum()
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("Overall Accuracy", f"{conf_accuracy*100:.1f}%")
            with perf_col2:
                st.metric("High Confidence Predictions", f"{high_conf}/{len(conf_report)}")
            with perf_col3:
                st.metric("Certain Predictions", f"{certain}/{len(conf_report)}")
            
            st.markdown(f"""
            **Key Insights:**
            - Model correctly predicted {(conf_report['Correct'].sum())}/{len(conf_report)} test cases
            - {((conf_report['Prediction_Type'] == 'Certain').sum()/len(conf_report)*100):.1f}% of predictions are certain (single class)
            - {((conf_report['Confidence_Level'] == 'High').sum()/len(conf_report)*100):.1f}% of predictions have high confidence
            """)
        
        # Model Download
        st.markdown("---")
        st.subheader("💾 Download Trained Models & Metadata")
        
        col1, col2, col3 = st.columns(3)
        
        # Base model
        with col1:
            pickle_buffer = BytesIO()
            pickle.dump(results['clf'], pickle_buffer)
            pickle_buffer.seek(0)
            
            st.download_button(
                label=f"📥 {params['model_name']}",
                data=pickle_buffer,
                file_name=f"{params['antibiotic'].replace(' ', '_')}_{params['model_name'].replace(' ', '_')}_model.pkl",
                mime="application/octet-stream",
                width='stretch',
                key="download_base_model"
            )
        
        # Conformal model (if trained)
        if params['use_conformal'] and CREPES_AVAILABLE and results['conformal_clf'] is not None:
            with col2:
                pickle_buffer_conf = BytesIO()
                pickle.dump(results['conformal_clf'], pickle_buffer_conf)
                pickle_buffer_conf.seek(0)
                
                st.download_button(
                    label="📥 Conformal Model",
                    data=pickle_buffer_conf,
                    file_name=f"{params['antibiotic'].replace(' ', '_')}_conformal_model.pkl",
                    mime="application/octet-stream",
                    width='stretch',
                    key="download_conformal_model"
                )
        
        # Metadata
        with col3:
            metadata = {
                'antibiotic': params['antibiotic'],
                'species': params['species'],
                'model_type': params['model_name'],
                'num_features': len(results['features']),
                'num_samples': len(results['y_test']),
                'num_classes': results['unique_classes'],
                'test_accuracy': float(results['accuracy']),
                'features': results['features'],
                'classes': list(results['class_names']),
            }
            
            json_buffer = BytesIO()
            json_buffer.write(json.dumps(metadata, indent=2).encode())
            json_buffer.seek(0)
            
            st.download_button(
                label="📋 Metadata",
                data=json_buffer,
                file_name=f"{params['antibiotic'].replace(' ', '_')}_metadata.json",
                mime="application/json",
                width='stretch',
                key="download_metadata"
            )