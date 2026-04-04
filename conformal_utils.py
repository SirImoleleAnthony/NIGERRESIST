# Conformal Prediction & Explainability Utilities for AMR Model Training


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import pickle

try:
    from crepes import WrapClassifier
    CREPES_AVAILABLE = True
except ImportError:
    CREPES_AVAILABLE = False


def load_antibiotic_data(antibiotic_name, use_cleaned=True):

    if use_cleaned:
        filepath = f"cleaned/{antibiotic_name}_Nigerian_subset_cleaned.csv"
    else:
        filepath = "cleaned/amr_nigeria_full_data.csv"
    
    try:
        df = pd.read_csv(filepath)
        if not use_cleaned:
            # Filter to specific antibiotic if using full data
            outcome_col = f"{antibiotic_name}_I"
            if outcome_col in df.columns:
                df = df[[col for col in df.columns if antibiotic_name not in col or col == outcome_col] + [outcome_col]]
        return df
    except FileNotFoundError:
        return None


def prepare_data_for_modeling(df, antibiotic_name, selected_species=None):
    
    # Filter by species if specified
    if selected_species:
        df = df[df['Species'] == selected_species].copy()
    else:
        selected_species = df['Species'].unique()[0] if 'Species' in df.columns else 'All'
    
    outcome_col = f"{antibiotic_name}_I"
    
    # Define features based on available columns
    # Target: {antibiotic_name}_I (Resistant, Susceptible, Intermediate)
    base_features = ['Study', 'Species', 'Family', 'Country', 'Gender', 'Age Group', 
                     'Speciality', 'Source', 'Year', 'Phenotype']
    
    # Include genetic features if available
    genetic_features = ['Genes', 'Genotype']
    for feat in genetic_features:
        if feat in df.columns:
            base_features.append(feat)
    
    features = [f for f in base_features if f in df.columns]
    
    # Encode categorical features
    encoding_dict = {}
    for col in features:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Other').astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoding_dict[col] = le
            elif col == 'Year':
                df[col] = df[col].fillna(df[col].median())
    
    # Drop rows with missing outcome
    df = df.dropna(subset=[outcome_col])
    
    # Encode outcome
    if df[outcome_col].dtype == 'object':
        le_outcome = LabelEncoder()
        y = le_outcome.fit_transform(df[outcome_col])
        y_original = df[outcome_col].values
    else:
        le_outcome = None
        y = df[outcome_col].values
        y_original = y
    
    X = df[features].copy()
    
    return {
        'X': X,
        'y': y,
        'y_original': y_original,
        'features': features,
        'le_outcome': le_outcome,
        'species': selected_species,
        'encoding_dict': encoding_dict
    }


def train_conformal_model(X_train, y_train, X_test, y_test, clf, alpha=0.1):

    if not CREPES_AVAILABLE:
        st.error("CREPES library not available. Install with: pip install crepes")
        return None, None, None, None, None
    
    # Train conformal predictor
    conformal_clf = WrapClassifier(clf)
    conformal_clf.fit(X_train, y_train)
    conformal_clf.calibrate(X_train, y_train)
    
    # Generate predictions with confidence intervals
    p_values = conformal_clf.predict_p(X_test)
    y_pred = conformal_clf.predict(X_test)
    
    # Calculate prediction sets (conformal sets)
    predicted_sets = []
    for scores in p_values:
        # Classes with p-value > alpha are in the prediction set
        predicted_set = [i for i, score in enumerate(scores) if score > alpha]
        predicted_sets.append(predicted_set)
    
    return conformal_clf, y_pred, predicted_sets, p_values




def create_conformal_report(y_pred, predicted_sets, p_values, y_test, le_outcome):

    report = pd.DataFrame({
        'Predicted_Class': [le_outcome.classes_[p] if le_outcome else p for p in y_pred],
        'Actual_Class': [le_outcome.classes_[y] if le_outcome else y for y in y_test],
        'Num_Classes_in_Set': [len(s) for s in predicted_sets],
        'Empty_Set': [len(s) == 0 for s in predicted_sets],
        'Single_Prediction': [len(s) == 1 for s in predicted_sets],
        'Max_Confidence': [max(scores) if len(scores) > 0 else 0 for scores in p_values]
    })
    
    report['Correct'] = report['Predicted_Class'] == report['Actual_Class']
    
    # Add interpretability columns
    report['Confidence_Level'] = report['Max_Confidence'].apply(
        lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.5 else 'Low')
    )
    
    report['Prediction_Type'] = report.apply(
        lambda row: 'Certain' if row['Single_Prediction'] else ('Uncertain' if row['Num_Classes_in_Set'] > 1 else 'Empty'),
        axis=1
    )
    
    return report
