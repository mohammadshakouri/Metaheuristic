FEATURE_CODE = "geo_direction_consistency_10"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Direction Consistency over 10 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    dir_consistency = delta.rolling(10, min_periods=10).apply(
        lambda x: np.prod(np.sign(x[x != 0])) if np.all(x != 0) else 0
    )

    s = dir_consistency.astype(float)
    s.name = FEATURE_CODE
    return s
