import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any

def create_attrition_gauge(probability: float) -> go.Figure:
    """Create attrition risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Attrition Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_risk_distribution_pie(risk_data: Dict[str, int]) -> go.Figure:
    """Create pie chart for risk distribution"""
    fig = px.pie(
        values=list(risk_data.values()),
        names=list(risk_data.keys()),
        title="Employee Risk Distribution",
        color=list(risk_data.keys()),
        color_discrete_map={
            'Low': 'lightgreen',
            'Medium': 'yellow',
            'High': 'red'
        }
    )
    return fig

def create_forecast_line_chart(forecast_df: pd.DataFrame) -> go.Figure:
    """Create line chart for skill demand forecast"""
    fig = px.line(
        forecast_df,
        x='date',
        y='forecasted_demand',
        title=f"Skill Demand Forecast: {forecast_df['skill_name'].iloc[0]}",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode='x unified'
    )
    
    return fig

def create_trending_skills_bar(trending_data: List[Dict]) -> go.Figure:
    """Create bar chart for trending skills"""
    df = pd.DataFrame(trending_data)
    
    fig = px.bar(
        df,
        x='skill_name',
        y='growth_rate',
        title="Trending Skills by Growth Rate",
        color='trend',
        color_discrete_map={
            'Rising': 'green',
            'Stable': 'gray',
            'Declining': 'red'
        }
    )
    
    fig.update_layout(
        xaxis_title="Skill",
        yaxis_title="Growth Rate (%)",
        xaxis_tickangle=-45
    )
    
    return fig

def create_skill_heatmap(employee_skills_df: pd.DataFrame) -> go.Figure:
    """Create heatmap of skills by department"""
    # Pivot data
    pivot = employee_skills_df.pivot_table(
        index='department',
        columns='skill_name',
        aggfunc='size',
        fill_value=0
    )
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Skill", y="Department", color="Count"),
        title="Skills Distribution by Department",
        aspect="auto"
    )
    
    return fig

def create_mobility_network(similar_employees: List[Dict]) -> go.Figure:
    """Create network visualization for similar employees"""
    # Simple scatter plot representation
    df = pd.DataFrame(similar_employees)
    
    fig = px.scatter(
        df,
        x=list(range(len(df))),
        y='similarity_score',
        text='employee_id',
        title="Similar Employees Network",
        size='similarity_score',
        size_max=20
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title="Employee Index",
        yaxis_title="Similarity Score",
        showlegend=False
    )
    
    return fig