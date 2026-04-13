import polars as pl
from catboost import CatBoostRegressor

from soilcast.data.aligned import AlignedDataFrame

feat_cont = [
    'YLDG',
    'FNO3yr', 'PRCPyr', 'TMXyr', 'TMNyr', 'RADyr', 'RSD',
    'SLP_PRC', 'SAND_TOP', 'CLAY_TOP', 'ELEV']

feat_cat = [
    'TILL', 'IRR', 'CROP'
]

def predict_rsd_all(
    model_rsdc: CatBoostRegressor, 
    model_rnad: CatBoostRegressor, 
    crop: int, 
    x_prod_pred: AlignedDataFrame, 
    y_prod_pred: AlignedDataFrame, 
    norm_params: pl.DataFrame, 
    hist: str = 'hist2', 
    ssp: str = '126') -> pl.DataFrame:

    crop_mean, crop_std = norm_params.filter(pl.col('CROP') == crop)[['YLDG_MEAN', 'YLDG_STD']].rows()[0]

    data_all = []
    for i in range(1, y_prod_pred.num_periods + 1):
        key = (i, hist if i <=2 else ssp)
        data_period = x_prod_pred[key]
        prod = y_prod_pred[key]['PROD']
        data_all.append(
            data_period.with_columns([
                pl.Series(prod * crop_std + crop_mean).alias('YLDG'),
                pl.lit(crop).alias('CROP')
            ])
        )

    data_pred = pl.concat(data_all, how='vertical')
    x_pred = data_pred.select(feat_cont + feat_cat).to_numpy()

    return data_pred[['PERIOD']].with_columns([
        pl.Series(model_rsdc.predict(x_pred)).alias('RSDCyr'),
        pl.Series(model_rnad.predict(x_pred)).alias('RNADyr')
    ])
