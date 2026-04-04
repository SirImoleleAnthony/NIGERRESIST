"""
Batch Prediction Interface for AMR Models

Allows users to:
1. Upload CSV files with multiple samples
2. Select trained model for batch predictions
3. Generate conformal prediction confidence intervals
4. Export results
5. Create epidemiological reports from batch results
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
from conformal_utils import (
    CREPES_AVAILABLE
)


def batch_predict_amr():
    """
    Batch prediction interface with CSV upload and conformal predictions.
    """
    
    st.title("📊 Batch Prediction & Epidemiological Analysis")
    
    st.markdown("""
    ### Process Multiple Isolates at Once
    
    This interface allows you to:
    1. **Upload CSV** with multiple bacterial isolates
    2. **Select trained model** to apply predictions
    3. **Get confidence intervals** for each prediction
    4. **Export comprehensive report** with explanations
    5. **Analyze trends** across batch results
    """)
    
    st.info("💡 **Tip:** Prepare your data with phenotypic/genotypic features matching your trained model.")
    
    # Step 1: Upload CSV
    st.subheader("1️⃣ Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Select CSV file with isolate data:",
        type=['csv'],
        help="CSV with columns matching trained model features"
    )
    
    if uploaded_file is None:
        st.warning("⚠️ No file uploaded. Please upload a CSV to proceed.")
        return
    
    # Load CSV
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return
    
    st.success(f"✅ Loaded {len(batch_df)} isolates with {len(batch_df.columns)} columns")
    
    # Show preview
    with st.expander("📋 Preview Data"):
        st.dataframe(batch_df.head(10), width='stretch')
    
    # Data info
    st.info(f"**Data Summary:** {len(batch_df)} rows × {len(batch_df.columns)} columns")
    
    # Step 2: Model Selection
    st.subheader("2️⃣ Select Trained Model")
    
    # List available models (from downloads/models if saved locally)
    model_dir = Path("saved_models")
    model_files = sorted([f.stem for f in model_dir.glob("*_model.pkl")]) if model_dir.exists() else []
    
    if not model_files:
        st.warning("⚠️ No saved models found. Train a model first in 'Train Model' page.")
        st.info("💾 **To use batch predictions:** Download and save model `.pkl` files to `saved_models/` folder")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        model_choice = st.selectbox(
            "Choose trained model:",
            model_files,
            help="Select model to use for batch predictions"
        )
    
    # Step 3: Load Model
    st.subheader("3️⃣ Batch Prediction Configuration")
    
    try:
        model_path = model_dir / f"{model_choice}_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success(f"✅ Loaded model: {model_choice}")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return
    
    # Load metadata if available
    metadata = None
    metadata_path = model_dir / f"{model_choice}_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            st.info(f"Model Info: {metadata['species']} → {metadata['antibiotic']}")
        except:
            pass
    
    # Feature selection and validation
    if metadata and 'features' in metadata:
        required_features = metadata['features']
        
        # Check if CSV has required features
        missing_features = [f for f in required_features if f not in batch_df.columns]
        if missing_features:
            st.error(f"❌ Missing columns: {missing_features}")
            st.info(f"✓ Required columns: {required_features}")
            return
        
        # Select features
        X_batch = batch_df[required_features].copy()
        st.success(f"✅ Found all {len(required_features)} required features")
    else:
        # If no metadata, let user select columns
        st.warning("⚠️ No metadata found. Manually select feature columns:")
        selected_cols = st.multiselect(
            "Select feature columns:",
            batch_df.columns,
            help="Choose columns that match model training features"
        )
        if not selected_cols:
            return
        X_batch = batch_df[selected_cols].copy()
    
    # Handle missing values
    col1, col2 = st.columns(2)
    with col1:
        missing_strategy = st.selectbox(
            "Handle missing values:",
            ["Drop rows", "Fill with median", "Fill with mode"],
            help="Strategy for handling NaN values"
        )
    
    with col2:
        st.metric("Missing values", X_batch.isnull().sum().sum())
    
    if X_batch.isnull().sum().sum() > 0:
        if missing_strategy == "Drop rows":
            X_batch = X_batch.dropna()
            st.info(f"Removed rows with missing values. {len(X_batch)} samples remain.")
        elif missing_strategy == "Fill with median":
            X_batch = X_batch.fillna(X_batch.median())
            st.info("Filled numeric columns with median value.")
        elif missing_strategy == "Fill with mode":
            X_batch = X_batch.fillna(X_batch.mode().iloc[0])
            st.info("Filled columns with mode value.")
    
    if len(X_batch) == 0:
        st.error("❌ No valid samples after preprocessing.")
        return
    
    # Step 4: Make Predictions
    st.subheader("4️⃣ Generate Predictions")
    
    if st.button("🚀 Predict for All Isolates", type="primary", width='stretch'):
        with st.spinner("Computing predictions for all isolates..."):
            
            # Make predictions
            try:
                y_pred = model.predict(X_batch)
                y_proba = model.predict_proba(X_batch) if hasattr(model, 'predict_proba') else None
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                return
            
            # Create results dataframe
            results_df = batch_df[len(batch_df) - len(X_batch):].copy()
            results_df['Predicted_Resistance'] = y_pred
            
            if y_proba is not None:
                # Add probabilities for each class
                classes = model.classes_ if hasattr(model, 'classes_') else np.unique(y_pred)
                for i, cls in enumerate(classes):
                    results_df[f'Probability_{cls}'] = y_proba[:, i]
                
                # Add confidence (max probability)
                results_df['Confidence'] = y_proba.max(axis=1)
            
            # Conformal predictions if available
            if CREPES_AVAILABLE:
                try:
                    # Try loading conformal model
                    conformal_path = model_dir / f"{model_choice.replace('_model', '')}_conformal_model.pkl"
                    if conformal_path.exists():
                        with open(conformal_path, 'rb') as f:
                            conformal_clf = pickle.load(f)
                        
                        # Get prediction sets
                        y_pred_conf, p_values = conformal_clf.predict(X_batch, return_p_values=True)
                        
                        # Count classes in prediction set
                        results_df['Prediction_Set_Size'] = [len(s) if s else 0 for s in y_pred_conf]
                        results_df['Prediction_Set'] = [str(s) for s in y_pred_conf]
                        results_df['P_Value'] = p_values.max(axis=1) if p_values is not None else 1.0
                        
                        st.success("✅ Conformal prediction sets computed")
                except:
                    st.info("ℹ️ Conformal model not found; using standard predictions")
            
            # Display results
            st.markdown("### Prediction Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(results_df))
            with col2:
                if 'Confidence' in results_df.columns:
                    avg_conf = results_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
            with col3:
                if 'Prediction_Set_Size' in results_df.columns:
                    avg_set_size = results_df['Prediction_Set_Size'].mean()
                    st.metric("Avg Set Size", f"{avg_set_size:.2f}")
            
            # Results table
            st.markdown("#### Detailed Results")
            display_cols = ['Predicted_Resistance']
            if 'Confidence' in results_df.columns:
                display_cols.append('Confidence')
            if 'Prediction_Set_Size' in results_df.columns:
                display_cols.append('Prediction_Set_Size')
            
            # Show all probability columns
            prob_cols = [col for col in results_df.columns if col.startswith('Probability_')]
            display_cols.extend(prob_cols)
            
            st.dataframe(results_df[display_cols], width='stretch')
            
            # Distribution visualization
            st.markdown("#### Resistance Distribution")
            
            col1, col2 = st.columns(2)
            
            # Prediction distribution
            with col1:
                pred_counts = results_df['Predicted_Resistance'].value_counts()
                fig_pred = px.bar(
                    x=pred_counts.index,
                    y=pred_counts.values,
                    labels={'x': 'Resistance Class', 'y': 'Count'},
                    title='Distribution of Predictions',
                    color=pred_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_pred, width='stretch')
            
            # Confidence distribution
            with col2:
                if 'Confidence' in results_df.columns:
                    fig_conf = px.histogram(
                        results_df,
                        x='Confidence',
                        nbins=20,
                        title='Prediction Confidence Distribution',
                        labels={'Confidence': 'Model Confidence'},
                        color_discrete_sequence=['#636EFA']
                    )
                    st.plotly_chart(fig_conf, width='stretch')
            
            # High/Low confidence samples
            if 'Confidence' in results_df.columns:
                st.markdown("#### High vs Low Confidence Predictions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    high_conf = len(results_df[results_df['Confidence'] >= 0.8])
                    st.metric("High Confidence (≥0.8)", f"{high_conf} ({high_conf/len(results_df)*100:.1f}%)")
                
                with col2:
                    low_conf = len(results_df[results_df['Confidence'] < 0.6])
                    st.metric("Low Confidence (<0.6)", f"{low_conf} ({low_conf/len(results_df)*100:.1f}%)")
                
                # Show low confidence samples
                if low_conf > 0:
                    st.warning(f"⚠️ {low_conf} predictions have low confidence. Review these:")
                    low_conf_df = results_df[results_df['Confidence'] < 0.6][[
                        'Predicted_Resistance', 'Confidence'
                    ] + prob_cols].head(10)
                    st.dataframe(low_conf_df, width='stretch')
            
            # Store results in session for download
            st.session_state.batch_results = results_df
            st.session_state.batch_model = model_choice
            
            st.success("✅ Predictions complete!")
    
    # Step 5: Export Results
    if 'batch_results' in st.session_state:
        st.markdown("---")
        st.subheader("5️⃣ Export Results")
        
        results_df = st.session_state.batch_results
        
        # Export formats
        col1, col2, col3 = st.columns(3)
        
        # CSV export
        with col1:
            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📥 CSV Results",
                data=csv_buffer.getvalue(),
                file_name=f"{st.session_state.batch_model}_batch_results.csv",
                mime="text/csv",
                width='stretch'
            )
        
        # Excel export (if openpyxl available)
        with col2:
            try:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Predictions', index=False)
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📥 Excel Report",
                    data=excel_buffer,
                    file_name=f"{st.session_state.batch_model}_batch_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch'
                )
            except:
                st.info("Excel export requires openpyxl")
        
        # JSON export
        with col3:
            json_str = results_df.to_json(orient='records', indent=2)
            
            st.download_button(
                label="📥 JSON Format",
                data=json_str,
                file_name=f"{st.session_state.batch_model}_batch_results.json",
                mime="application/json",
                width='stretch'
            )
        
        # Epidemiological Summary
        st.markdown("---")
        st.subheader("🔬 Epidemiological Analysis")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_samples = len(results_df)
            st.metric("Total Isolates", total_samples)
        
        with col2:
            if 'Predicted_Resistance' in results_df.columns:
                resistance_prevalence = (results_df['Predicted_Resistance'] == 'Resistant').sum() / len(results_df) * 100
                st.metric("Resistance Prevalence", f"{resistance_prevalence:.1f}%")
        
        with col3:
            if 'Confidence' in results_df.columns:
                avg_conf = results_df['Confidence'].mean()
                st.metric("Avg Model Confidence", f"{avg_conf*100:.1f}%")
        
        with col4:
            if 'Prediction_Set_Size' in results_df.columns:
                avg_set_size = results_df['Prediction_Set_Size'].mean()
                st.metric("Avg Pred Set Size", f"{avg_set_size:.2f}")
        
        # Generate epidemiological report
        st.markdown("### 📋 Summary Report")
        
        report = f"""
        **Batch Prediction Summary**
        - Total Isolates Analyzed: {len(results_df)}
        - Model Used: {st.session_state.batch_model}
        - Prediction Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        **Resistance Breakdown:**
        """
        
        if 'Predicted_Resistance' in results_df.columns:
            for cls in results_df['Predicted_Resistance'].unique():
                count = (results_df['Predicted_Resistance'] == cls).sum()
                pct = count / len(results_df) * 100
                report += f"\n        - {cls}: {count} ({pct:.1f}%)"
        
        if 'Confidence' in results_df.columns:
            report += f"""
        
        **Prediction Confidence Analysis:**
        - Mean: {results_df['Confidence'].mean():.3f}
        - Median: {results_df['Confidence'].median():.3f}
        - Min: {results_df['Confidence'].min():.3f}
        - Max: {results_df['Confidence'].max():.3f}
        """
        
        st.text(report)
        
        # Export report
        report_buffer = StringIO()
        report_buffer.write(report)
        report_buffer.seek(0)
        
        st.download_button(
            label="📄 Download Report",
            data=report_buffer.getvalue(),
            file_name=f"{st.session_state.batch_model}_epidemiological_report.txt",
            mime="text/plain",
            width='stretch'
        )


if __name__ == "__main__":
    batch_predict_amr()
