import pandas as pd
import streamlit as st
import altair as alt


def plot_location_forecast(df: pd.DataFrame):

    mapping = {
        "OCPD": ("Soil organic carbon [t/ha]", "#2ca02c", "SOC [t/ha]"),
        "TWN": ("Total nitrogen [t/ha]", "#1f77b4", "N [t/ha]"),
        "PROD": ("Productivity [normalized yield]", "#d62728", "Yield [norm]"),
    }

    charts = []

    for var, (title, color, ylabel) in mapping.items():

        d = df[["Period", var, f"{var}err"]].copy()
        d["lower"] = d[var] - d[f"{var}err"]
        d["upper"] = d[var] + d[f"{var}err"]

        base = alt.Chart(d).encode(
            x=alt.X(
                "Period:Q",
                title="Year",
                axis=alt.Axis(
                    values=[1,2,3,4,5,6],
                    labelExpr="""
                    datum.value == 1 ? '2000' :
                    datum.value == 2 ? '2020' :
                    datum.value == 3 ? '2040' :
                    datum.value == 4 ? '2060' :
                    datum.value == 5 ? '2080' :
                    '2100'
                    """
                )
            )
        )

        band = base.mark_area(
            color=color,
            opacity=0.2
        ).encode(
            y="lower:Q",
            y2="upper:Q"
        )

        line = base.mark_line(
            color=color,
            size=2
        ).encode(
            y=alt.Y(f"{var}:Q", title=ylabel)
        )

        chart = (band + line).properties(
            title=title,
            height=150
        )

        charts.append(chart)

    final_chart = alt.vconcat(*charts, spacing=30)

    st.altair_chart(final_chart, width=600)