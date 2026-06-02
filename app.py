from collections import UserDict
from pathlib import Path
from typing import Optional
from catboost import CatBoostRegressor
import joblib
import numpy as np
import polars as pl
import streamlit as st
import folium
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from streamlit_folium import st_folium
from soilcast.app import to_line_plot_data, to_map_plot_data, plot_location_forecast, plot_residue_forecast
from soilcast.model.ensemble import SoilCastModel
from soilcast.model.rsd import predict_rsd_all
from soilcast.data.aligned import AlignedDataFrame
from soilcast.app.inputs import cls_options, irr_options, till_options, ssp_options


base_dir = Path(__file__).resolve().parent
model_path = base_dir / 'models'
data_path = base_dir / 'data'

crop_labels = {
    1: "Corn",
    2: "Potato",
    3: "Rapeseed",
    4: "Rice",
    5: "Spring barley",
    6: "Sugar beet",
    7: "Soybean",
    8: "Sunflower",
    9: "Winter rye",
    10: "Winter wheat",
    11: "Clover",
    12: "Cotton",
    13: "Corn silage",
    14: "Durum wheat",
    15: "Flax seed",
    16: "Field pea",
    17: "Oats",
}

sim_years = [2000, 2020, 2040, 2060, 2080, 2100]

st.set_page_config(
    page_title="AI4SoilHealth Soil Health Projection",
    layout="centered"
)

# Default values
if "mgt_type" not in st.session_state:
    st.session_state.mgt_type = "Static"

for yr in [None] + sim_years:
    suffix = f"_{yr}" if yr else ""

    if f"till{suffix}" not in st.session_state:
        st.session_state[f"till{suffix}"] = "conv"

    if f"irr{suffix}" not in st.session_state:
        st.session_state[f"irr{suffix}"] = "rf"

    if f"cls{suffix}" not in st.session_state:
        st.session_state[f"cls{suffix}"] = 1

    if f"ftn{suffix}" not in st.session_state:
        st.session_state[f"ftn{suffix}"] = 0

    if f"use_bau{suffix}" not in st.session_state:
        st.session_state[f"use_bau{suffix}"] = True

    if f"rsd{suffix}" not in st.session_state:
        st.session_state[f"rsd{suffix}"] = 0

if "ssp" not in st.session_state:
    st.session_state.ssp = "126"

if "mode" not in st.session_state:
    st.session_state.mode = None

def selector(label, field, options):
    val = st.selectbox(
        label, 
        options=options.keys(), 
        format_func=lambda x: options[x],
        width="stretch",
        key=field
    )
    return val

def toggle(label, field, options):
    val = st.segmented_control(
        label, 
        options=options.keys(), 
        format_func=lambda x: options[x],
        width="stretch",
        key=field
    )
    return val

@st.cache_resource
def init_model() -> SoilCastModel:
    return SoilCastModel.load(model_path)

@st.cache_resource
def init_models_rsd() -> dict[str, CatBoostRegressor]:
    return {x: joblib.load(model_path / f"{x}.p") for x in ["RSDCyr", "RNADyr"]}

@st.cache_resource
def load_baseline_data() -> AlignedDataFrame:
    df = pl.read_parquet(data_path / "climsoil.parquet")
    return AlignedDataFrame(df, keys=["TILL", "IRR", "CLS", "RSD", "FTN"])

def show_management(yr: Optional[int] = None):
    
    suffix = f"_{yr}" if yr else ""

    selector("Crop rotation", f"cls{suffix}", cls_options)
    toggle("Tillage", f"till{suffix}", till_options)
    toggle("Irrigation", f"irr{suffix}", irr_options)
    
    col1, col2 = st.columns([3, 1])

    with col2:
        st.checkbox("BAU", key=f"use_bau{suffix}")

    with col1:
        st.slider(
            "Fertilizer [kg/ha]",
            min_value=0.0,
            max_value=250.0,
            disabled=st.session_state[f"use_bau{suffix}"],
            step=1.0,
            key=f"ftn{suffix}"
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        rsd = st.slider(
            "Residue retention [%]",
            min_value=0,
            max_value=90,
            step=1,
            format="%d%%",
            key=f"rsd{suffix}"
        )

def get_scenario_dict(yr: Optional[int] = None) -> dict:
    suffix = f"_{yr}" if yr else ""
    scenario = {
        "TILL": st.session_state[f"till{suffix}"],
        "IRR": st.session_state[f"irr{suffix}"],
        "CLS": st.session_state[f"cls{suffix}"],
        "RSD": st.session_state[f"rsd{suffix}"],
    }
    if not st.session_state[f"use_bau{suffix}"]:
        scenario["FNO3yr"] = st.session_state[f"ftn{suffix}"]
    return scenario

def setup_x_pred(data: AlignedDataFrame) -> AlignedDataFrame:

    new = AlignedDataFrame.__new__(AlignedDataFrame)
    UserDict.__init__(new)

    if st.session_state.mgt_type == "Static":
        values = get_scenario_dict()
        for key, df in data.items():
            new.data[key] = df.with_columns(
                [pl.lit(v).cast(df.schema[col]).alias(col) for col, v in values.items()]
            )

    elif st.session_state.mgt_type == "Dynamic":
        for key, df in data.items():
            yr = {i_: x for i_, x in enumerate(sim_years)}[key[0] - 1]
            values = get_scenario_dict(yr)
            new.data[key] = df.with_columns(
                [pl.lit(v).cast(df.schema[col]).alias(col) for col, v in values.items()]
            )

    else:
        raise ValueError("Invalid management type")

    return new

with st.sidebar:

    st.header("Forecast Parameters", divider=True)
    st.subheader("Field Management")

    tab_static, tab_dyn = st.tabs(["Static", "Dynamic"], key="mgt_type", on_change="rerun")

    with tab_static:
        show_management()
    
    with tab_dyn:
        for yr, tab in zip(sim_years, st.tabs([str(x) for x in sim_years])):
            with tab:
                show_management(yr)

    st.subheader("Scenario")
    toggle("Climate projection", "ssp", ssp_options)

    mode = st.segmented_control(
        "Run simulation",
        ["Single location", "Pan-European"],
        key="mode",
        width="stretch"
    )

placeholder = st.empty()

if st.session_state.mode is not None:
    placeholder = st.empty()

    model = init_model()
    models_rsd = init_models_rsd()
    yldg_norm_params = pl.read_parquet(data_path / "yldg_norm.parquet")
    data = load_baseline_data()

    if mode == "Single location":
        st.title("Single Location Forecast")

        with st.expander("Pick location", expanded=True):

            location = st.session_state.get("location", [50, 10])

            if "map_center" not in st.session_state:
                st.session_state.map_center = location

            if "map_zoom" not in st.session_state:
                st.session_state.map_zoom = 4

            m = folium.Map(
                location=st.session_state.map_center, 
                zoom_start=st.session_state.map_zoom
            )

            if "location" in st.session_state:
                icon = folium.CustomIcon(
                    "assets/marker-icon.png",
                    icon_size=(25, 41),
                    icon_anchor=(12, 41),
                )
                
                folium.Marker(
                    st.session_state.location, 
                    icon=icon
                ).add_to(m)

            map_data = st_folium(m, height=500, width=700)

            if map_data and map_data.get("last_clicked"):
                lat = map_data["last_clicked"]["lat"]
                lon = map_data["last_clicked"]["lng"]

                st.session_state.location = (lat, lon)
                st.session_state.map_center = [
                    map_data["center"]["lat"],
                    map_data["center"]["lng"]
                ]
                st.session_state.map_zoom = map_data["zoom"]
                st.rerun()

        if "location" in st.session_state:
            lat, lon = st.session_state.location
            data_location, dist = data.find_nearest_simu(lat, lon)
            df_location = data_location[(1, 'hist2')]

            with st.container(border=True):
                st.subheader(f"Your location: {lat:.2f}, {lon:.2f}")
                st.text("Climate and soil data for the forecast will be taken from the nearest simulation unit in our data.")
                c1, c2 = st.columns([2, 1], width=350, gap=None)
                with c1:
                    st.text("Distance to next centroid")
                    st.text("Initial carbon in topsoil")
                    st.text("Initial nitrogen in topsoil")
                with c2:
                    st.markdown(f"**{(dist / 1000):.1f} km**")
                    st.markdown(f"**{df_location['OCPDinit'].item():.1f} t/ha**")
                    st.markdown(f"**{df_location['TWNinit'].item():.1f} t/ha**")

            # data_location = data_location.with_columns_static(scenario)
            data_location = setup_x_pred(data_location)

            y_pred = model.predict(data_location, start_ssp="hist2")

            with st.container(border=True):
                st.subheader("Soil Health & Productivity Forecast")
                plot_data = to_line_plot_data(y_pred, data_path / "error.p")
                plot_data = plot_data[plot_data["SSP"] == st.session_state.ssp]
                plot_location_forecast(plot_data)
                
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.subheader("Carbon in Residues")
                    
                with c2:
                    selected_crop = st.selectbox(
                        "Assumed crop",
                        options=list(crop_labels),
                        index=0,
                        format_func=lambda crop: crop_labels[crop],
                        key="rsd_crop",
                    )

                rsd_pred = predict_rsd_all(
                    model_rsdc=models_rsd["RSDCyr"],
                    model_rnad=models_rsd["RNADyr"],
                    crop=selected_crop,
                    x_prod_pred=data_location,
                    y_prod_pred=y_pred,
                    norm_params=yldg_norm_params,
                    hist="hist2",
                    ssp=st.session_state.ssp
                )
                plot_residue_forecast(rsd_pred)


    elif mode == "Pan-European":
        st.title("Pan-European Forecast")

        col1, col2 = st.columns(2)

        with col1:
            st.selectbox("Year", options=sim_years, key="eu_sim_year")

        with col2:
            selector(
                "Sample size", 
                "sample", 
                options={1000: "low (1k)", 20000: "moderate (20k)", 40000: "high (40k)"}
            )

        data_eu = setup_x_pred(data)

        if st.session_state.sample < 87277:
            data_eu = data_eu.sample(st.session_state.sample)
        
        y_pred = model.predict(data_eu, start_ssp="hist2")

        period_idx = (st.session_state.eu_sim_year - 2000) // 20 + 1
        if period_idx > 2:
            key = (period_idx, st.session_state.ssp)
        else:
            key = (period_idx, "hist2")

        y_pred = to_map_plot_data(data_eu, y_pred)
        df = y_pred[key]

        fig, axs = plt.subplots(
            3, 1,
            figsize=(10, 30),
            subplot_kw={"projection": ccrs.PlateCarree()},
            layout="compressed"
        )

        for i, (response, title, ylabel, cmap) in enumerate([
            ("OCPD", "Soil organic carbon", "Soil organic carbon [t/ha]", "summer_r"),
            ("TWN", "Total nitrogen", "Total nitrogen [t/ha]", "autumn_r"),
            ("PROD", "Productivity", "Productivity [normalized]", "winter_r")
        ]):

            ax = axs[i]

            # Europe extent
            ax.set_extent([-12, 40, 34, 72], crs=ccrs.PlateCarree())

            # basemap
            ax.add_feature(cfeature.LAND, facecolor="white")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            ax.add_feature(cfeature.OCEAN, facecolor="white")

            sc = ax.hexbin(
                df["lon"],
                df["lat"],
                C=df[response],
                reduce_C_function=np.mean,
                gridsize=120,
                cmap=cmap,
                transform=ccrs.PlateCarree()
            )

            ax.set_title(title)

            # colorbar
            cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label(ylabel)

        st.pyplot(fig)

else:
    with placeholder.container():
        st.title("Welcome To SoilCast!")
        st.text("""This application allows you to explore the impacts of agricultural management and climate scenarios on soil and crop dynamics.

    You can run simulations for a specific location or explore pan-European projections under different management practices and climate pathways. Adjust field management parameters in the sidebar and run the simulation to generate forecasts.

    Use the Single Location mode to analyse conditions for a specific field site, or switch to Europe-wide mode to examine broader spatial patterns.
    """)
        st.write("")
        st.write("")
        st.write("")
        st.image(base_dir / "assets" / "ai4sh.svg")
        st.write("")
        st.write("")
        st.write("")
        st.markdown(
            "*SoilCast is a research prototype developed for interactive exploration of soil and crop simulation scenarios. " +
            "Predictions are generated by machine-learning surrogate models trained on simulation data and should be interpreted " +
            "as exploratory model outputs rather than operational forecasts.*")
