from collections import UserDict
import numpy as np
import polars as pl
from pyproj import Transformer


class AlignedDataFrame(UserDict):

    def __init__(self, stacked: pl.DataFrame, keys: list[str]):
        super().__init__()
        self._initialized = False
        self._align(stacked, keys)
        self._initialized = True
        

    def with_columns(self, key_values: dict) -> "AlignedDataFrame":
        new = AlignedDataFrame.__new__(AlignedDataFrame)
        UserDict.__init__(new)

        for key, df in self.items():
            new.data[key] = df.with_columns(
                [pl.lit(v).alias(col) for col, v in key_values.items()]
            )

        return new
    
    def find_nearest_simu(self, lat, lon) -> tuple["AlignedDataFrame", float]:
        transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3035", always_xy=True
        )
        lon, lat = transformer.transform(lon, lat)

        df = next(iter(self.values()))

        dist = (df["YLAT"] - lat)**2 + (df["XLONG"] - lon)**2
        idx = dist.arg_min()
        simu_id = df["SimUID"][idx]

        new = AlignedDataFrame.__new__(AlignedDataFrame)
        UserDict.__init__(new)

        for k, v in self.items():
            new.data[k] = v.filter(pl.col("SimUID") == simu_id)

        return new, float(dist[idx] ** 0.5)
    
    def sample(self, size: int, seed=None):
        rng = np.random.default_rng(seed)

        first_df = next(iter(self.values()))
        n = first_df.height

        subset_idx = rng.choice(n, size=min(size, n), replace=False)

        new = AlignedDataFrame.__new__(AlignedDataFrame)
        UserDict.__init__(new)

        for k, df in self.items():
            new.data[k] = df[subset_idx]

        return new
    
    @staticmethod
    def validate(data: "AlignedDataFrame"):
        """
        Validate that blocks[(PERIOD, SSP)] contain perfectly aligned entity_id rows.

        Checks:
        - identical row counts
        - identical ordered entity_id column
        - unique (PERIOD, SSP) keys
        """

        if not data:
            raise ValueError("Data is empty")

        seen = set()
        base_entity_ids = None
        base_rows = None

        for (period, ssp), df in data.items():

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

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls.__new__(cls)
        UserDict.__init__(obj)

        obj.data = dict(data)

        return obj

    @property
    def height(self) -> int:
        if len(self) > 0:
            return self[list(self.keys())[0]].height
        else:
            return 0

    @property
    def num_periods(self) -> int:
        return max([x[0] for x in self.keys()])

    def _align(self, df: pl.DataFrame, keys: list[str]):
        entities = (
            df.select(keys)
            .unique()
            .sort(keys)
            .with_row_index("entity_id")
        )
        df = df.join(entities, on=keys, how="inner")
        blocks = df.partition_by(["PERIOD", "SSP"], maintain_order=True)
        for b in blocks:
            self[(b["PERIOD"][0], b["SSP"][0])] = b.sort("entity_id")

    def __setitem__(self, key, value):
        if self._initialized:
            raise TypeError("Read-only")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        if self._initialized:
            raise TypeError("Read-only")

def align_by_keys(df: pl.DataFrame, keys: list) -> pl.DataFrame:
    """
    Returns a DF that only contains rows with matching management across all periods. 
    This is used in validation only.
    """
    n_blocks = (
        df.select(pl.struct(["PERIOD","SSP"]).n_unique())
        .item()
    )

    valid_entities = (
        df.group_by(keys)
        .agg(pl.struct(["PERIOD","SSP"]).n_unique().alias("n"))
        .filter(pl.col("n") == n_blocks)
        .select(keys)
    )

    aligned = df.join(valid_entities, on=keys)
    return aligned
