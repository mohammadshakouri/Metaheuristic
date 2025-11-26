FEATURE_CODE = "info_return_entropy_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Information Return Entropy over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    returns = g["close"].pct_change().rolling(20, min_periods=20).apply(
        lambda x: -np.sum((x + 1) * np.log(x + 1 + 1e-10)) / np.log(len(x))
    )

    s = returns.astype(float)
    s.name = FEATURE_CODE
    return s
