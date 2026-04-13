from pathlib import Path
import warnings
import numpy as np
import polars as pl


cols_internal = ['SimUID', 'SSP', 'FTN', 'HC', 'PRODinit', 'YLAT', 'XLONG']

def load_all(path_hc1: Path, path_hc2: Path, columns: list[str], separate_hist: bool = False) -> pl.DataFrame:
    """
    Loads and merges consolidated HC1 and HC2 parquet dataframes.
    """
    data_hc1 = pl.read_parquet(path_hc1)
    data_hc2 = pl.read_parquet(path_hc2)

    data_hc1 = data_hc1.with_columns([
        pl.col('RES').str.slice(1).cast(pl.UInt8).alias('RSD'),
        pl.lit('conv').alias('TILL'),
        pl.lit('rf').alias('IRR'),
        pl.lit(None).cast(pl.String).alias('SSP'),
        pl.lit(1).cast(pl.Int8).alias('HC')
    ])

    data_hc2 = data_hc2.with_columns([
        pl.col('RES').str.slice(1).cast(pl.UInt8).alias('RSD'),
        pl.lit(2).cast(pl.Int8).alias('HC')
    ])

    # Remove HC2 outliers in mintill
    data_hc2 = pl.concat([
        data_hc2.filter(pl.col('TILL') == 'conv'),
        remove_outliers_iqr('OCPDd', data_hc2.filter(pl.col('TILL') == 'mintill'))
    ], how='vertical')

    data_hc2 = data_hc2.filter(pl.col('OCPDinit') < 250)

    data = pl.concat([
            data_hc1[columns], 
            data_hc2[columns]
        ], how='vertical')
    
    if not separate_hist:
        data = data.with_columns(
            pl.col("SSP").fill_null("hist")
        )
    else:
        data = data.with_columns(
            pl.coalesce(
                pl.col("SSP"),
                pl.when(pl.col("HC") == 1).then(pl.lit("hist1"))
                .when(pl.col("HC") == 2).then(pl.lit("hist2"))
            ).alias("SSP")
        )

    return data

def remove_outliers_iqr(yname: str, data: pl.DataFrame):
    '''
    Removes outliers as definded by the whiskers of a boxplot.
    '''
    q1 = np.percentile(data[yname], 25, method='midpoint')
    q3 = np.percentile(data[yname], 75, method='midpoint')
    assert not np.isnan(q1) and not np.isnan(q3)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    data_filtered = data.filter((pl.col(yname) >= lower) & (pl.col(yname) <= upper))
    num_removed = data.height - data_filtered.height

    warnings.warn(f'Removed {(num_removed)} ({(num_removed / data.height):.2f}%) outliers in {yname}')
    return data_filtered