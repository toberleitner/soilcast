from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import polars as pl
import pyproj


responses = ['OCPDd', 'TWNd', 'PRODd']
feat_cont = [
    'OCPDinit', 'TWNinit', 
    'FNO3yr', 'PRCPyr', 'TMXyr', 'TMNyr', 'RADyr', 'RSD',
    'SLP_PRC', 'SAND_TOP', 'CLAY_TOP', 'ELEV'
]
feat_cat = ['CLS', 'TILL', 'IRR', 'PERIOD']
cols_internal = ['SimUID', 'SSP', 'FTN', 'HC', 'PRODinit', 'YLAT', 'XLONG']

def cubify(df: pl.DataFrame, keys: list) -> dict[tuple[int, str], pl.DataFrame]:

    entities = (
        df.select(keys)
        .unique()
        .sort(keys)
        .with_row_index("entity_id")
    )

    df = df.join(entities, on=keys, how="inner")

    blocks = df.partition_by(["PERIOD", "SSP"], maintain_order=True)

    blocks = {
        (b["PERIOD"][0], b["SSP"][0]): b.sort("entity_id")
        for b in blocks
    }

    return blocks

def force_scenario(data: dict[tuple[int, str], pl.DataFrame], scenario: dict):
    return {
        key: df.with_columns(
            [pl.lit(v).alias(col) for col, v in scenario.items()]
        )
        for key, df in data.items()
    }

def validate_cube(blocks: dict[tuple[int, str], pl.DataFrame]) -> None:
    """
    Validate that blocks[(PERIOD, SSP)] contain perfectly aligned entity_id rows.

    Checks:
    - identical row counts
    - identical ordered entity_id column
    - unique (PERIOD, SSP) keys
    """

    if not blocks:
        raise ValueError("Blocks dictionary is empty")

    seen = set()
    base_entity_ids = None
    base_rows = None

    for (period, ssp), df in blocks.items():

        if (period, ssp) in seen:
            raise ValueError(f"Duplicate block detected for {(period, ssp)}")
        seen.add((period, ssp))

        if "entity_id" not in df.columns:
            raise ValueError(f"Block {(period, ssp)} missing 'entity_id' column")

        entity_ids = df.get_column("entity_id")

        if base_entity_ids is None:
            base_entity_ids = entity_ids
            base_rows = df.height
            continue

        if df.height != base_rows:
            raise ValueError(
                f"Row count mismatch in block {(period, ssp)}: "
                f"{df.height} != {base_rows}"
            )

        if not entity_ids.equals(base_entity_ids):
            raise ValueError(
                f"Entity alignment mismatch in block {(period, ssp)}"
            )

def find_nearest_simu(data, lat, lon):
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)
    lon, lat = transformer.transform(lon, lat)

    df =  data[list(data)[0]]
    dist = (df["YLAT"] - lat)**2 + (df["XLONG"] - lon)**2
    idx = dist.arg_min()
    simu_id = df[idx]['SimUID'].item()

    return {k: v.filter(pl.col('SimUID') == simu_id) for k, v in data.items()}, np.sqrt(dist[idx])

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

    err_df = pd.DataFrame(error).loc[['medae']].T
    err_df.index.names = ['Period', 'SSP', 'Response']

    df = df.merge(err_df.loc[:, :, 'OCPDd'].rename({'medae': 'OCPDerr'}, axis=1), on=['Period', 'SSP'], how='left')
    df = df.merge(err_df.loc[:, :, 'TWNd'].rename({'medae': 'TWNerr'}, axis=1), on=['Period', 'SSP'], how='left')
    df = df.merge(err_df.loc[:, :, 'PRODd'].rename({'medae': 'PRODerr'}, axis=1), on=['Period', 'SSP'], how='left')

    df = df.sort_index()

    errcols = ['OCPDerr', 'TWNerr', 'PRODerr']
    df[errcols] = df.groupby(level="SSP")[errcols].cumsum()

    df = df.reset_index()

    return df

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

def sample(data, size: int):
    """
    Perform sampling on aligned data.
    """
    rng = np.random.default_rng(42)
    n = data[list(data.keys())[0]].height
    subset_idx = rng.choice(n, size=min(size, n), replace=False)

    data_sample = {}
    for k in data.keys():
        data_sample[k] = data[k][subset_idx]

    return data_sample