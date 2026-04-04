"""
Model Comparison Dashboard for AMR Predictions

Allows users to:
1. Train multiple models on same data
2. Compare performance metrics
3. Visualize accuracy, precision, recall, F1
4. Compare conformal prediction coverage
5. Analyze feature importance differences
6. Export comparison report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from io import BytesIO, StringIO
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)

from conformal_utils import (
    load_antibiotic_data, 
    prepare_data_for_modeling,
    train_conformal_model,
    CREPES_AVAILABLE
)


def compare_models():
    """
    Train and compare multiple models on the same data.
    """
    
    st.title("⚖️ Model Comparison Dashboard")
    
    st.markdown("""
    ### Compare Multiple Algorithms
    
    This interface allows you to:
    1. **Train multiple models** on the same dataset
    2. **Compare performance metrics** (Accuracy, Precision, Recall, F1, ROC-AUC)
    3. **Visualize results** with interactive charts
    4. **Analyze feature importance** across models
    5. **Compare conformal prediction** coverage
    6. **Export detailed report** with rankings
    """)
    
    # Step 1: Data Selection
    st.subheader("1️⃣ Select Data for Comparison")
    
    # Get available antibiotics
    from pathlib import Path
    cleaned_dir = Path("cleaned")
    antibiotic_files = sorted([
        f.name.replace("_Nigerian_subset_cleaned.csv", "") 
        for f in cleaned_dir.glob("*_Nigerian_subset_cleaned.csv")
    ])
    
    if not antibiotic_files:
        st.error("❌ No cleaned datasets found.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_antibiotic = st.selectbox(
            "Select antibiotic:",
            antibiotic_files,
            help="Choose antibiotic-specific dataset"
        )
    
    # Load data
    df = load_antibiotic_data(selected_antibiotic, use_cleaned=True)
    if df is None:
        st.error(f"Could not load {selected_antibiotic}")
        return
    
    with col2:
        species_list = sorted(df['Species'].unique().tolist())
        selected_species = st.selectbox(
            "Select species:",
            species_list,
            help="Choose bacterial species"
        )
    
    # Prepare data
    data_prep = prepare_data_for_modeling(df, selected_antibiotic, selected_species)
    X = data_prep['X']
    y = data_prep['y']
    features = data_prep['features']
    le_outcome = data_prep['le_outcome']
    
    if len(X) < 20:
        st.warning(f"⚠️ Only {len(X)} samples (need ≥20)")
        return
    
    unique_classes = len(np.unique(y))
    if unique_classes < 2:
        st.warning("⚠️ Only 1 class detected")
        return
    
    st.success(f"✅ {len(X)} samples | {len(features)} features | {unique_classes} classes")
    
    # Step 2: Model Selection
    st.subheader("2️⃣ Select Models to Compare")
    
    all_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(objective='multi:softmax', num_class=unique_classes, 
                                eval_metric='mlogloss', random_state=42, verbosity=0),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    
    selected_models = st.multiselect(
        "Choose models to compare:",
        list(all_models.keys()),
        default=['Random Forest', 'Gradient Boosting', 'XGBoost'],
        help="Select at least 2 models for meaningful comparison"
    )
    
    if len(selected_models) < 2:
        st.warning("⚠️ Select at least 2 models for comparison")
        return
    
    # Step 3: Comparison Settings
    st.subheader("3️⃣ Comparison Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider(
            "Test size:",
            0.1, 0.4, 0.2, 0.05
        )
    
    with col2:
        n_runs = st.slider(
            "Number of runs:",
            1, 5, 1,
            help="Train each model multiple times with different splits for robustness"
        )
    
    with col3:
        use_conformal = st.checkbox(
            "Compare conformal",
            value=CREPES_AVAILABLE and CREPES_AVAILABLE,
            disabled=not CREPES_AVAILABLE
        )
    
    # Run comparison
    if st.button("🚀 Train and Compare Models", type="primary", width='stretch'):
        
        with st.spinner("Training and comparing models..."):
            
            # Store results
            comparison_results = []
            model_artifacts = {}
            
            # For each run
            for run_num in range(n_runs):
                progress_text = f"Run {run_num + 1}/{n_runs}"
                progress_bar = st.progress(0, text=progress_text)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42 + run_num, stratify=y
                )
                
                # Train each model
                for idx, model_name in enumerate(selected_models):
                    progress_pct = (idx + 1) / len(selected_models)
                    progress_bar.progress(progress_pct, text=f"{progress_text}: {model_name}")
                    
                    # Train model
                    clf = all_models[model_name]
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    
                    # Compute metrics
                    metrics = {
                        'Model': model_name,
                        'Run': run_num + 1,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    }
                    
                    # ROC-AUC for binary classification
                    if unique_classes == 2:
                        try:
                            y_proba = clf.predict_proba(X_test)[:, 1]
                            metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba)
                        except:
                            metrics['ROC-AUC'] = 0.0
                    
                    # Conformal prediction
                    if use_conformal and CREPES_AVAILABLE:
                        try:
                            conf_clf, _, pred_sets, _ = train_conformal_model(
                                X_train, y_train, X_test, y_test, clf, alpha=0.1
                            )
                            if pred_sets:
                                coverage = (y_test.values[None, :] == np.array([pred_sets]).T).any(axis=1).mean()
                                avg_set_size = np.mean([len(s) if s else 0 for s in pred_sets])
                                metrics['Conformal_Coverage'] = coverage
                                metrics['Avg_Set_Size'] = avg_set_size
                        except:
                            pass
                    
                    comparison_results.append(metrics)
                    
                    # Store first model for artifacts
                    if run_num == 0:
                        model_artifacts[model_name] = clf
            
            progress_bar.empty()
            
            # Results dataframe
            results_df = pd.DataFrame(comparison_results)
            
            # Aggregate across runs
            agg_results = results_df.groupby('Model').agg({
                'Accuracy': ['mean', 'std'],
                'Precision': ['mean', 'std'],
                'Recall': ['mean', 'std'],
                'F1': ['mean', 'std'],
            }).round(4)
            
            if 'ROC-AUC' in results_df.columns:
                agg_results[('ROC-AUC', 'mean')] = results_df.groupby('Model')['ROC-AUC'].mean()
                agg_results[('ROC-AUC', 'std')] = results_df.groupby('Model')['ROC-AUC'].std()
            
            # Store in session
            st.session_state.comparison_results = results_df
            st.session_state.agg_results = agg_results
            st.session_state.selected_antibiotic = selected_antibiotic
            st.session_state.selected_species = selected_species
            st.session_state.model_artifacts = model_artifacts
            
            st.success("✅ Comparison complete!")
    
    # Display results
    if 'comparison_results' in st.session_state:
        
        results_df = st.session_state.comparison_results
        agg_results = st.session_state.agg_results
        
        st.markdown("---")
        st.subheader("📊 Comparison Results")
        
        # Overall rankings
        st.markdown("#### 🏆 Model Rankings (Average Metrics)")
        
        # Flatten multi-index for display
        rankings = pd.DataFrame({
            'Model': agg_results.index,
            'Accuracy': agg_results[('Accuracy', 'mean')].values,
            'Precision': agg_results[('Precision', 'mean')].values,
            'Recall': agg_results[('Recall', 'mean')].values,
            'F1': agg_results[('F1', 'mean')].values,
        })
        
        if ('ROC-AUC', 'mean') in agg_results.columns:
            rankings['ROC-AUC'] = agg_results[('ROC-AUC', 'mean')].values
        
        # Sort by F1
        rankings = rankings.sort_values('F1', ascending=False)
        
        # Add ranks
        rankings.insert(0, 'Rank', range(1, len(rankings) + 1))
        
        st.dataframe(rankings.reset_index(drop=True), width='stretch')
        
        # Visualization
        st.markdown("#### Performance Comparison")
        
        # Metrics to visualize
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
        if 'ROC-AUC' in results_df.columns:
            metrics_to_plot.append('ROC-AUC')
        
        # Create comparison chart
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            if metric in results_df.columns:
                metric_data = results_df.groupby('Model')[metric].mean().sort_values(ascending=False)
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metric_data.index,
                    y=metric_data.values,
                    text=[f"{v:.3f}" for v in metric_data.values],
                    textposition='auto'
                ))
        
        fig.update_layout(
            title=f"Model Performance Comparison: {st.session_state.selected_species} → {st.session_state.selected_antibiotic}",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Individual metrics breakdown
        with st.expander("📈 Detailed Metrics Breakdown"):
            cols = st.columns(len(metrics_to_plot))
            for col_idx, metric in enumerate(metrics_to_plot):
                if metric in results_df.columns:
                    with cols[col_idx]:
                        metric_data = results_df.groupby('Model')[metric].apply(list)
                        
                        metric_stats = pd.DataFrame({
                            'Model': metric_data.index,
                            'Mean': [np.mean(v) for v in metric_data.values],
                            'Std': [np.std(v) for v in metric_data.values],
                            'Min': [np.min(v) for v in metric_data.values],
                            'Max': [np.max(v) for v in metric_data.values],
                        }).sort_values('Mean', ascending=False)
                        
                        st.markdown(f"**{metric}**")
                        st.dataframe(metric_stats, width='stretch')
        
        # Feature importance comparison (for tree-based models)
        st.markdown("---")
        st.subheader("🌳 Feature Importance Comparison")
        
        tree_models = {k: v for k, v in st.session_state.model_artifacts.items() 
                       if hasattr(v, 'feature_importances_')}
        
        if tree_models:
            selected_tree_model = st.selectbox(
                "Select model for feature importance:",
                list(tree_models.keys())
            )
            
            if selected_tree_model:
                clf = tree_models[selected_tree_model]
                fi_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': clf.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig_fi = px.bar(
                    fi_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'Top 10 Features: {selected_tree_model}',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_fi, width='stretch')
        
        # Conformal prediction comparison
        if 'Conformal_Coverage' in results_df.columns:
            st.markdown("---")
            st.subheader("🎯 Conformal Prediction Comparison")
            
            conf_summary = results_df.groupby('Model').agg({
                'Conformal_Coverage': 'mean',
                'Avg_Set_Size': 'mean'
            }).round(3)
            
            st.dataframe(conf_summary, width='stretch')
            
            # Visualization
            fig_conf = px.scatter(
                conf_summary.reset_index(),
                x='Avg_Set_Size',
                y='Conformal_Coverage',
                text='Model',
                title='Coverage vs Prediction Set Size',
                hover_name='Model',
                color_discrete_sequence=['#636EFA']
            )
            fig_conf.update_traces(textposition='top center', marker=dict(size=15))
            st.plotly_chart(fig_conf, width='stretch')
        
        # Export comparison
        st.markdown("---")
        st.subheader("💾 Export Comparison Report")
        
        col1, col2, col3 = st.columns(3)
        
        # CSV
        with col1:
            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📥 Raw Results CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{st.session_state.selected_antibiotic}_{st.session_state.selected_species}_comparison.csv",
                mime="text/csv",
                width='stretch'
            )
        
        # Summary statistics
        with col2:
            summary_buffer = StringIO()
            summary_buffer.write(f"Model Comparison Report\n")
            summary_buffer.write(f"{'='*50}\n\n")
            summary_buffer.write(f"Antibiotic: {st.session_state.selected_antibiotic}\n")
            summary_buffer.write(f"Species: {st.session_state.selected_species}\n")
            summary_buffer.write(f"Samples: {len(X)}\n")
            summary_buffer.write(f"Features: {len(features)}\n")
            summary_buffer.write(f"Classes: {unique_classes}\n\n")
            summary_buffer.write(f"{'RANKINGS'}\n")
            summary_buffer.write(f"{'-'*50}\n")
            summary_buffer.write(rankings.to_string())
            summary_buffer.seek(0)
            
            st.download_button(
                label="📄 Summary Report",
                data=summary_buffer.getvalue(),
                file_name=f"{st.session_state.selected_antibiotic}_{st.session_state.selected_species}_summary.txt",
                mime="text/plain",
                width='stretch'
            )
        
        # JSON
        with col3:
            json_results = {
                'metadata': {
                    'antibiotic': st.session_state.selected_antibiotic,
                    'species': st.session_state.selected_species,
                    'samples': len(X),
                    'features': len(features),
                    'classes': unique_classes
                },
                'results': results_df.to_dict(orient='records')
            }
            json_str = json.dumps(json_results, indent=2)
            
            st.download_button(
                label="📋 JSON Report",
                data=json_str,
                file_name=f"{st.session_state.selected_antibiotic}_{st.session_state.selected_species}_comparison.json",
                mime="application/json",
                width='stretch'
            )


if __name__ == "__main__":
    compare_models()
