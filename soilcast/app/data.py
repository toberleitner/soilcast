from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import polars as pl
import pyproj


def to_line_plot_data(y_pred, error_file_path: Path) -> pd.DataFrame:
    
    with open(error_file_path, 'rb') as f:
        error = pickle.load(f)

    df = {}
    for ssp in ['126', '585']:

        df[ssp] = pd.DataFrame(
            [
                {k2: v2[0] for k2, v2 in v.items()}
                for k, v in y_pred.items() 
                if k[1] in ['hist2', ssp]
            ], 
            index=np.arange(1, 6 + 1)
        )
    df = pd.concat(df)
    df.index.names = 'SSP', 'Period'

    err_df = pd.DataFrame(error).loc[['quantiles']].T
    err_df.index.names = ['Period', 'SSP', 'Response']
    err_df['low'] = err_df['quantiles'].apply(lambda x: x[0])
    err_df['high'] = err_df['quantiles'].apply(lambda x: x[1])
    err_df = err_df.drop('quantiles', axis=1)

    df_and_err = df.copy()

    for response in ['OCPD', 'TWN', 'PROD']:
        for band in ['low', 'high']:
            df_and_err = df_and_err.merge(
                err_df.loc[:, :, f'{response}d'].rename({band: response + band}, axis=1)[response + band], on=['Period', 'SSP'], 
                how='left'
            )

    df_and_err = df_and_err.sort_index().reset_index()

    return df_and_err

def to_map_plot_data(baseline_data, y_pred):
    transformer = pyproj.Transformer.from_crs(
        "EPSG:3035", "EPSG:4326", always_xy=True
    )
    
    merged = {}
    for k in baseline_data.keys():
        df = baseline_data[k].select(['SimUID', 'YLAT', 'XLONG'])

        lon, lat = transformer.transform(
            df['XLONG'].to_numpy(),
            df['YLAT'].to_numpy()
        )

        merged[k] = (
            df
            .select(['SimUID'])
            .with_columns([
                pl.Series("lon", lon),
                pl.Series("lat", lat),
                *[
                    pl.Series(r, y_pred[k][r])
                    for r in ['OCPD', 'TWN', 'PROD', 'OCPDd', 'TWNd', 'PRODd']
                ]
            ])
        )

    return merged
