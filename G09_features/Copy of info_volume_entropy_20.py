FEATURE_CODE = "info_volume_entropy_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Information Volume Entropy over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    vol_changes = g["volume"].pct_change().rolling(20, min_periods=20).apply(
        lambda x: -np.sum((x + 1) * np.log(x + 1 + 1e-10)) / np.log(len(x))
    )

    s = vol_changes.astype(float)
    s.name = FEATURE_CODE
    return s
