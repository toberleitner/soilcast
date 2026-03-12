import numpy as np
from catboost import CatBoostRegressor


def forecast(
    blocks: dict[tuple[int, str]],
    model_ocpd: CatBoostRegressor,
    model_twn: CatBoostRegressor,
    model_prod: CatBoostRegressor,
    feat_cont: list,
    feat_cat: list,
    start_period: int = 1,
    start_ssp: str = "hist1"
):
    """
    Fast recursive prediction over aligned blocks.

    blocks: dict[(PERIOD, SSP)] -> DataFrame
    """

    # --- determine period order
    periods = sorted({p for p, _ in blocks})

    # --- initialize from first block
    base = blocks[(start_period, start_ssp)]

    ocpd = base["OCPDinit"].to_numpy().copy()
    twn = base["TWNinit"].to_numpy().copy()
    prod = base["PRODinit"].to_numpy().copy()

    results = {}

    for period in periods:

        # determine available SSPs for this period
        ssps = [s for (p, s) in blocks if p == period]

        for ssp in ssps:

            df = blocks[(period, ssp)]

            # --- continuous features
            X_cont = df.select(feat_cont).to_numpy()

            # overwrite recursive features
            X_cont[:, 0] = ocpd
            X_cont[:, 1] = twn

            # --- categorical features
            X_cat = df.select(feat_cat).to_numpy()

            # --- full model input
            X = np.hstack([X_cont, X_cat])

            # --- predictions
            ocpd_d = model_ocpd.predict(X)
            twn_d = model_twn.predict(X)
            prod_d = model_prod.predict(X)

            # --- recursive update
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

    return results

