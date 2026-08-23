import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

from soilcast.models.ensemble import SoilCastModel
from soilcast.data.aligned import AlignedDataFrame


def error_quantiles(y_true, y_pred, q_low=0.05, q_high=0.95):
    errors = y_pred - y_true
    return (
        np.quantile(errors, q_low),
        np.quantile(errors, q_high)
    )

def validate(model: SoilCastModel, data_test: AlignedDataFrame):

    y_pred_all = model.predict(data_test)
    eval_results = {}

    for (period, ssp), data in data_test.items():
        for response in SoilCastModel.responses:
            y_true = data.select(response)
            y_pred = y_pred_all[(period, ssp)][response]

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            medae = median_absolute_error(y_true, y_pred)
            abs_err_quants = error_quantiles(y_true.to_numpy().ravel(), y_pred)

            eval_results[(period, ssp, response)] = {
                'r2': r2, 
                'rmse': rmse, 
                'mae': mae, 
                'medae': medae,
                'quantiles': abs_err_quants
            }

    return eval_results
