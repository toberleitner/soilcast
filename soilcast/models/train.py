import json
from pathlib import Path
import time
import warnings
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from catboost import CatBoostRegressor
import logging


def train_model_err_adaptive(
        data_train: pl.DataFrame, data_test: pl.DataFrame, 
        feat_cont: list[str], feat_cat: list[str], response: str, 
        logger: logging.Logger):
    """
    Uses error-driven adaptive sampling to train a catboost model.
    """

    initial_sample = 200_000
    add_per_round = 100_000
    rounds = 2
    probe_size = 1_000_000

    if response == 'PRODd':
        rounds = 4

    rng = np.random.default_rng(42)

    X_full = data_train[feat_cont + feat_cat].to_pandas()
    y_full = data_train[response].to_numpy()

    N = len(X_full)

    # Initial set
    subset_idx = rng.choice(N, size=min(initial_sample, N), replace=False)

    for r in range(rounds):

        logger.info(f"Sampling round {r+1}")

        X_sub = X_full.iloc[subset_idx]
        y_sub = y_full[subset_idx]

        model = CatBoostRegressor(
            iterations=5000,
            max_depth=8,
            learning_rate=0.1,
            cat_features=feat_cat,
            task_type="GPU",
            devices="0",
            verbose=True
        )

        model.fit(X_sub, y_sub)

        probe_idx = rng.choice(N, size=min(probe_size, N), replace=False)
        X_probe = X_full.iloc[probe_idx]
        y_probe = y_full[probe_idx]

        pred = model.predict(X_probe)
        residuals = np.abs(pred - y_probe)

        # Select hardest samples
        worst = probe_idx[np.argsort(residuals)[-add_per_round:]]
        subset_idx = np.unique(np.concatenate([subset_idx, worst]))

        logger.info(f"Subset size now: {len(subset_idx)}")

    # Final training set
    X_train = X_full.iloc[subset_idx]
    y_train = y_full[subset_idx]

    X_test = data_test[feat_cont + feat_cat].to_pandas()
    y_test = data_test[response].to_numpy()

    logger.info(f"Final training size: {len(X_train)}")

    # Final model
    model = CatBoostRegressor(
        iterations=20_000,
        max_depth=8,
        learning_rate=0.1,
        cat_features=feat_cat,
        task_type='GPU',
        devices='0',
        od_type="Iter",
        od_wait=30,
        metric_period=50,
        use_best_model=True,
        verbose=True
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    return (
        model,
        {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'medae': medae
        }
    )

def spatial_split(data: pl.DataFrame, train_frac: float = 0.9):

    np.random.seed(42)

    # Find SUs that exist in all periods.
    simus_all_periods = (
        data
        .group_by("SimUID")
        .agg(pl.col("PERIOD").n_unique().alias("n_periods"))
        .filter(pl.col("n_periods") == 6)
        .get_column("SimUID")
        .sort()
        .sample(fraction=1, shuffle=True, seed=42)
    )
    n_total = len(simus_all_periods)
    n_train = int(train_frac * n_total)
    n_test = (n_total - n_train) // 2

    train_ids = simus_all_periods[:n_train].implode()
    test_ids = simus_all_periods[n_train : (n_train + n_test)].implode()
    validate_ids = simus_all_periods[(n_train + n_test):].implode()

    return train_ids, test_ids, validate_ids

def train_all(data: pl.DataFrame, feat_cont: list[str], feat_cat: list[str], responses: list[str], model_path: Path, logger: logging.Logger):

    train_ids, test_ids, validate_ids = spatial_split(data)

    train_df = data.filter(pl.col("SimUID").is_in(train_ids))
    test_df = data.filter(pl.col("SimUID").is_in(test_ids))

    # Record training, test and validation sets
    with open(model_path / 'training.json', 'w') as f:
        json.dump({
            'train_ids': train_ids.to_list()[0],
            'test_ids': test_ids.to_list()[0],
            'validate_ids': validate_ids.to_list()[0],
        }, f, indent=4)

    for response in responses:
        logger.info(f'Processing response {response}')

        start_time = time.time()
        model, stats = train_model_err_adaptive(train_df, test_df, feat_cont, feat_cat, response, logger)
        duration = time.time() - start_time

        filename = f'{response}'

        try:
            with open(model_path / (filename + '.json'), 'w') as f:
                json.dump(stats, f, indent=4)
        except:
            warnings.warn(f"Couldn't store f'{filename}.json'")

        try:
            model.save_model(model_path / (filename + '.cbm'))
        except:
            warnings.warn(f"Couldn't store f'{filename}.cbm'")

    logger.info('done')
