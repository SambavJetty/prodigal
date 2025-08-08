import altair as alt
import pandas as pd

def metrics_chart(df: pd.DataFrame):
    """
    df columns: metric (str), value (float)
    """
    return (
        alt.Chart(df)
        .mark_bar(size=40)
        .encode(
            x=alt.X("metric:N", title="Metric"),
            y=alt.Y("value:Q", title="Percent", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("metric:N", legend=None),
            tooltip=["metric", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=220)
    )
