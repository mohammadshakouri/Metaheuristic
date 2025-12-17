FEATURE_CODE = "mom_inertia_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Inertia over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    inertia = delta.rolling(30, min_periods=30).apply(lambda x: np.sum(x**2))

    s = inertia.astype(float)
    s.name = FEATURE_CODE
    return s
