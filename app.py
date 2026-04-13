from pathlib import Path
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
from soilcast.app.inputs import UserInput, cls_options, irr_options, till_options, ssp_options


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

st.set_page_config(
    page_title="AI4SoilHealth Soil Health Projection",
    layout="centered"
)

if "user_input" not in st.session_state:
    st.session_state.user_input = UserInput()

if "fert_bau" not in st.session_state:
    st.session_state.fert_bau = True

if "run_sim" not in st.session_state:
    st.session_state.run_sim = False

def update(field, value):
    setattr(st.session_state.user_input, field, value)

def selector(label, field, options):
    val = st.selectbox(
        label, 
        options=options.keys(), 
        format_func=lambda x: options[x],
        width='stretch'
    )
    update(field, val)

def toggle(label, field, options):
    val = st.segmented_control(
        label, 
        options=options.keys(), 
        format_func=lambda x: options[x],
        default=list(options.keys())[0], 
        width='stretch'
    )
    update(field, val)

@st.cache_resource
def init_model() -> SoilCastModel:
    return SoilCastModel.load(model_path)

@st.cache_resource
def init_models_rsd() -> dict[str, CatBoostRegressor]:
    return {x: joblib.load(model_path / f'{x}.p') for x in ['RSDCyr', 'RNADyr']}

@st.cache_resource
def load_baseline_data() -> AlignedDataFrame:
    df = pl.read_parquet(data_path / 'climsoil.parquet')
    return AlignedDataFrame(df, keys=["TILL", "IRR", "CLS", "RSD", "FTN"])

with st.sidebar:

    with st.form("simulation_form", border=False):

        st.header('Forecast Parameters', divider=True)
        st.subheader('Field Management')
        
        selector('Crop rotation', 'cls', cls_options)
        toggle('Tillage', 'till', till_options)
        toggle('Irrigation', 'irr', irr_options)

        col1, col2 = st.columns([3, 1])

        with col2:
            st.checkbox("BAU", key='fert_bau')

        with col1:
            st.slider(
                "Fertilizer [kg/ha]",
                min_value=0.0,
                max_value=250.0,
                value=0.01,
                step=1.0,
                key='fert_value'
            )

        col1, col2 = st.columns([3, 1])
        with col1:
            rsd = st.slider(
                "Residue retention [%]",
                min_value=0,
                max_value=90,
                value=0,
                step=1,
                format="%d%%"
            )
            update('rsd', rsd)

        st.subheader('Scenario')
        toggle('Climate projection', 'ssp', ssp_options)

        mode = st.segmented_control(
            'Simulation scope',
            ['Single location', 'Pan-European'],
            default='Single location',
            key='mode',
            width='stretch'
        )

        if st.form_submit_button('Run simulation', type='primary', width='stretch'):
            st.session_state.run_sim = True

placeholder = st.empty()

if st.session_state.run_sim:
    placeholder = st.empty()

    if st.session_state.fert_bau:
        st.session_state.user_input.ftn = None
    else:
        st.session_state.user_input.ftn = st.session_state.fert_value

    model = init_model()
    models_rsd = init_models_rsd()
    yldg_norm_params = pl.read_parquet(data_path / 'yldg_norm.parquet')
    data = load_baseline_data()

    scenario = {
        'TILL': st.session_state.user_input.till,
        'IRR': st.session_state.user_input.irr,
        'CLS': st.session_state.user_input.cls,
        'RSD': st.session_state.user_input.rsd,
    }
    if st.session_state.user_input.ftn is not None:
        scenario['FNO3yr'] = st.session_state.fert_value

    if mode == 'Single location':
        st.title('Single Location Forecast')

        with st.expander('Pick location', expanded=True):

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
                st.subheader(f'Your location: {lat:.2f}, {lon:.2f}')
                st.text('Climate and soil data for the forecast will be taken from the nearest simulation unit in our data.')
                c1, c2 = st.columns([2, 1], width=350, gap=None)
                with c1:
                    st.text('Distance to next centroid')
                    st.text('Initial carbon in topsoil')
                    st.text('Initial nitrogen in topsoil')
                with c2:
                    st.markdown(f'**{(dist / 1000):.1f} km**')
                    st.markdown(f"**{df_location['OCPDinit'].item():.1f} t/ha**")
                    st.markdown(f"**{df_location['TWNinit'].item():.1f} t/ha**")

            data_location = data_location.with_columns(scenario)

            y_pred = model.predict(data_location, start_ssp='hist2')

            with st.container(border=True):
                st.subheader('Soil Health & Productivity Forecast')
                plot_data = to_line_plot_data(y_pred, data_path / 'error.p')
                plot_data = plot_data[plot_data['SSP'] == st.session_state.user_input.ssp]
                plot_location_forecast(plot_data)

                
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.subheader('Carbon in Residues')
                    
                with c2:
                    selected_crop = st.selectbox(
                        'Assumed crop',
                        options=list(crop_labels),
                        index=0,
                        format_func=lambda crop: crop_labels[crop],
                        key='rsd_crop',
                    )

                rsd_pred = predict_rsd_all(
                    model_rsdc=models_rsd['RSDCyr'],
                    model_rnad=models_rsd['RNADyr'],
                    crop=selected_crop,
                    x_prod_pred=data_location,
                    y_prod_pred=y_pred,
                    norm_params=yldg_norm_params,
                    hist='hist2',
                    ssp=st.session_state.user_input.ssp
                )
                plot_residue_forecast(rsd_pred)


    elif mode == 'Pan-European':
        st.title('Pan-European Forecast')
        st.slider('Select target year', 2000, 2100, value=2000, step=20, key='eu_sim_year')
        st.slider('Sample size', 100, 87277, value=100, key='sample')

        data_eu = data.with_columns(scenario)

        if st.session_state.sample < 87277:
            data_eu = data_eu.sample(st.session_state.sample)
        
        y_pred = model.predict(data_eu, start_ssp='hist2')

        period_idx = (st.session_state.eu_sim_year - 2000) // 20 + 1
        if period_idx > 2:
            key = (period_idx, st.session_state.user_input.ssp)
        else:
            key = (period_idx, 'hist2')

        y_pred = to_map_plot_data(data_eu, y_pred)
        df = y_pred[key]

        fig, axs = plt.subplots(
            3, 1,
            figsize=(10, 30),
            subplot_kw={"projection": ccrs.PlateCarree()},
            layout="compressed"
        )

        for i, (response, title, ylabel, cmap) in enumerate([
            ('OCPD', 'Soil organic carbon', 'Soil organic carbon [t/ha]', 'summer_r'),
            ('TWN', 'Total nitrogen', 'Total nitrogen [t/ha]', 'autumn_r'),
            ('PROD', 'Productivity', 'Productivity [normalized]', 'winter_r')
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
        st.title('Welcome To SoilCast!')
        st.text("""This application allows you to explore the impacts of agricultural management and climate scenarios on soil and crop dynamics.

    You can run simulations for a specific location or explore pan-European projections under different management practices and climate pathways. Adjust field management parameters in the sidebar and run the simulation to generate forecasts.

    Use the Single Location mode to analyse conditions for a specific field site, or switch to Europe-wide mode to examine broader spatial patterns.
    """)
        st.write("")
        st.write("")
        st.write("")
        st.image(base_dir / 'assets' / 'ai4sh.svg')
        st.write("")
        st.write("")
        st.write("")
        st.markdown(
            '*SoilCast is a research prototype developed for interactive exploration of soil and crop simulation scenarios. ' +
            'Predictions are generated by machine-learning surrogate models trained on simulation data and should be interpreted ' +
            'as exploratory model outputs rather than operational forecasts.*')
