FEATURE_CODE = "vol_dynamic_range_index_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volatility Dynamic Range Index over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high_low_range = g["high"] - g["low"]
    dri = high_low_range.rolling(30, min_periods=30).apply(
        lambda x: (x.max() - x.min()) / x.mean() if x.mean() != 0 else np.nan
    )

    s = dri.astype(float)
    s.name = FEATURE_CODE
    return s
