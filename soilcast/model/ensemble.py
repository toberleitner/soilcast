
from pathlib import Path
import joblib
import numpy as np
from catboost import CatBoostRegressor
from soilcast.data.aligned import AlignedDataFrame


class SoilCastModel:

    responses = ['OCPDd', 'TWNd', 'PRODd']
    feat_cont = [
        'OCPDinit', 'TWNinit', 
        'FNO3yr', 'PRCPyr', 'TMXyr', 'TMNyr', 'RADyr', 'RSD',
        'SLP_PRC', 'SAND_TOP', 'CLAY_TOP', 'ELEV'
    ]
    feat_cat = ['CLS', 'TILL', 'IRR', 'PERIOD']

    def __init__(self,  model_ocpd: CatBoostRegressor, model_twn: CatBoostRegressor, model_prod: CatBoostRegressor):
        self.model_ocpd = model_ocpd
        self.model_twn = model_twn
        self.model_prod = model_prod    

    def predict(
        self,
        x_pred: AlignedDataFrame,
        start_period: int = 1,
        start_ssp: str = "hist1",
        prod_relative: bool = True
    ) -> AlignedDataFrame:
        """
        Fast recursive prediction over aligned blocks.

        blocks: dict[(PERIOD, SSP)] -> DataFrame
        """
        periods = sorted({p for p, _ in x_pred})
        base = x_pred[(start_period, start_ssp)]
        ocpd = base["OCPDinit"].to_numpy().copy()
        twn = base["TWNinit"].to_numpy().copy()

        if prod_relative:
            prod = np.zeros(x_pred.height)
        else:
            prod = base["PRODinit"].to_numpy().copy()

        results = {}

        results[(0, start_ssp)] = {
            "OCPD": ocpd.copy(),
            "TWN": twn.copy(),
            "PROD": prod.copy(),
            "OCPDd": np.zeros_like(ocpd),
            "TWNd": np.zeros_like(ocpd),
            "PRODd": np.zeros_like(ocpd),
        }

        for period in periods:

            for ssp in [s for (p, s) in x_pred if p == period]:
                df = x_pred[(period, ssp)]
                X_cont = df.select(self.feat_cont).to_numpy()
                X_cont[:, 0] = ocpd
                X_cont[:, 1] = twn
                X_cat = df.select(self.feat_cat).to_numpy()
                X = np.hstack([X_cont, X_cat])

                ocpd_d = self.model_ocpd.predict(X)
                twn_d = self.model_twn.predict(X)
                prod_d = self.model_prod.predict(X)

                ocpd = ocpd + ocpd_d
                twn = twn + twn_d
                prod = prod + prod_d

                results[(period, ssp)] = {
                    "OCPD": ocpd.copy(),
                    "TWN": twn.copy(),
                    "PROD": prod.copy(),
                    "OCPDd": ocpd_d,
                    "TWNd": twn_d,
                    "PRODd": prod_d,
                }

        return AlignedDataFrame.from_dict(results)

    @classmethod
    def load(cls, model_path: Path) -> "SoilCastModel":
        cat_models = {x: joblib.load(model_path / f'{x}.p') for x in cls.responses}
        return SoilCastModel(
            model_ocpd=cat_models['OCPDd'], 
            model_twn=cat_models['TWNd'], 
            model_prod=cat_models['PRODd']
        )