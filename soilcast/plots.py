import pandas as pd
import streamlit as st
import altair as alt


def plot_location_forecast(df: pd.DataFrame):

    df_long = df.melt(
        id_vars=["Period", "SSP"],
        value_vars=["OCPD", "TWN", "PROD"],
        var_name="variable",
        value_name="value"
    )

    df_err = df.melt(
        id_vars=["Period", "SSP"],
        value_vars=["OCPDerr", "TWNerr", "PRODerr"],
        var_name="variable",
        value_name="err"
    )

    df_err["variable"] = df_err["variable"].str.replace("err", "")

    df_plot = df_long.merge(df_err, on=["Period", "SSP", "variable"])

    df_plot["variable_label"] = df_plot["variable"].map({
        "OCPD": "Soil organic carbon",
        "TWN": "Total nitrogen",
        "PROD": "Productivity"
    })

    base = (
        alt.Chart(df_plot)
        .transform_calculate(
            lower="datum.value - datum.err",
            upper="datum.value + datum.err"
        )
        .encode(
            x=alt.X(
                "Period:Q",
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
            ),
            color=alt.Color(
                "variable_label:N",
                legend=alt.Legend(orient="right", title=None)
            )
        )
    )

    band_left = (
        base.transform_filter("datum.variable != 'PROD'")
        .mark_area(opacity=0.2)
        .encode(
            y=alt.Y("lower:Q", axis=None),
            y2="upper:Q"
        )
    )

    band_right = (
        base.transform_filter("datum.variable == 'PROD'")
        .mark_area(opacity=0.2)
        .encode(
            y=alt.Y("lower:Q", axis=None),
            y2="upper:Q"
        )
    )

    line_left = (
        base.transform_filter("datum.variable != 'PROD'")
        .mark_line()
        .encode(
            y=alt.Y("value:Q", title="C / N [kg/ha]")
        )
    )

    line_right = (
        base.transform_filter("datum.variable == 'PROD'")
        .mark_line(strokeDash=[4,2])
        .encode(
            y=alt.Y(
                "value:Q",
                title="Productivity [normalized yield]",
                axis=alt.Axis(orient="right")
            )
        )
    )

    chart = (
        alt.layer(
            band_left,
            band_right,
            line_left,
            line_right
        )
        .resolve_scale(y="independent")
        .properties(height=350, padding={"bottom": 30})
    )

    st.altair_chart(chart)