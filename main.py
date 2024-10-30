import streamlit as st
import pandas as pd
import numpy as np
from utils.model import train_model, prepare_data
from utils.visualizations import plot_confusion_matrix, plot_feature_importance, plot_probability_distribution
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None

# Title and description
st.title("Customer Churn Prediction Dashboard")
st.markdown("""
This application helps predict customer churn using logistic regression.
Upload your customer data to get started!
""")

# File upload section
st.header("1. Data Upload")
uploaded_file = st.file_uploader(
    "Upload your CSV file (required columns: ContractLength, MonthlyCharges, Churn)",
    type=['csv']
)

if uploaded_file is not None:
    try:
        # Load and preprocess data
        df = pd.read_csv(uploaded_file)
        X, y = prepare_data(df)
        
        # Train model and get predictions
        st.header("2. Model Training and Evaluation")
        with st.spinner("Training model..."):
            model, metrics, predictions, feature_importance = train_model(X, y)
            st.session_state['model'] = model
            st.session_state['predictions'] = predictions
            st.session_state['metrics'] = metrics

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")

        # Visualizations
        st.header("3. Visualizations")
        
        # Two-column layout for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            conf_matrix_fig = plot_confusion_matrix(
                metrics['confusion_matrix']
            )
            st.plotly_chart(conf_matrix_fig, use_container_width=True)
            
        with col2:
            st.subheader("Feature Importance")
            feature_importance_fig = plot_feature_importance(
                feature_importance,
                ['ContractLength', 'MonthlyCharges']
            )
            st.plotly_chart(feature_importance_fig, use_container_width=True)

        # Probability distribution
        st.subheader("Churn Probability Distribution")
        prob_dist_fig = plot_probability_distribution(predictions)
        st.plotly_chart(prob_dist_fig, use_container_width=True)

        # Download predictions
        st.header("4. Download Predictions")
        predictions_df = pd.DataFrame({
            'CustomerID': range(len(predictions)),
            'Churn_Probability': predictions
        })
        
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")
