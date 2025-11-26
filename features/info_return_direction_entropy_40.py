FEATURE_CODE = "info_return_direction_entropy_40"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Information Return Direction Entropy over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    dir_changes = (delta > 0).astype(int) - (delta < 0).astype(int)

    direction_entropy = dir_changes.rolling(40, min_periods=40).apply(
        lambda x: -np.sum((x.value_counts(normalize=True) * np.log(x.value_counts(normalize=True) + 1e-10)))
    )

    s = direction_entropy.astype(float)
    s.name = FEATURE_CODE
    return s
