FEATURE_CODE = "mom_persistence_index_25"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Persistence Index over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    persistence = delta.rolling(25, min_periods=25).apply(
        lambda x: np.sum(x > 0) / 25
    )

    s = persistence.astype(float)
    s.name = FEATURE_CODE
    return s
