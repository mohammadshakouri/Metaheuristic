FEATURE_CODE = "dyn_trend_reversal_index_40"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trend Reversal Index over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    rev_up = delta.clip(lower=0).rolling(40, min_periods=40).sum()
    rev_down = -delta.clip(upper=0).rolling(40, min_periods=40).sum()

    tri = rev_down / rev_up.replace(0, np.nan)

    s = tri.astype(float)
    s.name = FEATURE_CODE
    return s
