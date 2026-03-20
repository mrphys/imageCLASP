import plotly.express as px
import plotly.graph_objects as go



my_palette = [
    px.colors.qualitative.Plotly[9],
    px.colors.qualitative.Plotly[5],
    px.colors.qualitative.Plotly[2],
    px.colors.qualitative.Plotly[3],
    px.colors.qualitative.Plotly[4],
    px.colors.qualitative.Plotly[5],
    px.colors.qualitative.Plotly[6],
    px.colors.qualitative.Plotly[7],
    px.colors.qualitative.Plotly[8],
]


def hist(df, var, bin_start, bin_end, bin_size, colors):
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
    fig.update_layout(margin=dict(t=10), height=300)
    return fig



def pie(df, var, colors):
    counts = df[var].fillna("Unknown").value_counts().reset_index()
    counts.columns = [var, "count"]

    if var == "patient_sex":
        counts[var] = counts[var].replace({
            "M": "Male",
            "F": "Female"
        })

    if var == "DL_processed" or var == "roundel_processed":
        counts[var] = counts[var].replace({
            True: "Yes",
            False: "No"
        })
        
    fig = px.pie(counts, names=var, values="count")
    fig.update_traces(textposition="inside", textinfo="percent+label",textfont=dict(size=14,family="Helvetica"))
    fig.update_traces(marker_colors=colors)
    fig.update_layout(showlegend=False, margin=dict(t=10), height=300)
    return fig



