import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
my_palette = [
    '#317c63',
    "#5d2866",
    "#2B356F",
    "#222D6C",
]


def hist(df, var, colors, title):
    x=df[var].dropna()
    bin_start = (np.min(x) // 10) * 10
    bin_end = (np.max(x) // 10) * 10
    bin_size = (bin_end-bin_start)/5
    fig = go.Figure(
        go.Histogram(
            x=df[var].dropna(),
            xbins=dict(start=bin_start, end=bin_end, size=bin_size),
            marker_color=colors,
        )
    )
    fig.update_xaxes(range=[bin_start, bin_end])
    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=18)
        ),
        showlegend=False,
        margin=dict(t=40),
        height=300
    )
    return fig


def pie(df, var, colors):
    print(df)
    counts = df[var].fillna("Unknown").value_counts().reset_index()
    counts.columns = [var, "count"]
        
    fig = px.pie(counts, names=var, values="count")
    
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=14, family="Helvetica"),
        marker_colors=colors
    )
    
    fig.update_layout(
        title=dict(
            text=var.replace("_", " ").title(),
            x=0.5,
            xanchor="center",
            font=dict(size=18)
        ),
        showlegend=False,
        margin=dict(t=40),
    )
    
    return fig

def pie_from_counts(pos, total, label):
    df = pd.DataFrame({
        "label": [label, f"Not {label}"],
        "count": [pos, total - pos]
    })
    
    fig = px.pie(df, names="label", values="count")
    
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker_colors=my_palette
    )
    
    fig.update_layout(
        title=dict(
            text=label.title(),
            x=0.5,
            xanchor="center",
            font=dict(size=18)
        ),
        showlegend=False,
        margin=dict(t=40),
        height=300
    )
    return fig

def st_header(title="CLASP dashboard"):
    st.markdown(f"""
    <style>
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    .custom-title {{
        text-align: center;
        margin-bottom: 70px;
        font-size: 40px;
        font-family: Arial, sans-serif;
        font-weight: bold;
    }}
    </style>

    <h1 class="custom-title">{title}</h1>
    """, unsafe_allow_html=True)