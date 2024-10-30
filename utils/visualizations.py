import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_confusion_matrix(conf_matrix):
    """Create confusion matrix visualization."""
    labels = ['Not Churned', 'Churned']
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=labels,
        y=labels,
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=400,
        height=400
    )
    
    return fig

def plot_feature_importance(importance, feature_names):
    """Create feature importance visualization."""
    fig = go.Figure(data=go.Bar(
        x=feature_names,
        y=importance,
        text=np.round(importance, 3),
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Absolute Coefficient Value',
        width=400,
        height=400
    )
    
    return fig

def plot_probability_distribution(probabilities):
    """Create probability distribution visualization."""
    fig = go.Figure(data=go.Histogram(
        x=probabilities,
        nbinsx=30,
        name='Probability Distribution'
    ))
    
    fig.update_layout(
        title='Distribution of Churn Probabilities',
        xaxis_title='Churn Probability',
        yaxis_title='Count',
        showlegend=False
    )
    
    return fig
