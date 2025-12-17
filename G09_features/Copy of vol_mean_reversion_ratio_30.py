FEATURE_CODE = "vol_mean_reversion_ratio_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Mean Reversion Ratio over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    mean_price = g["close"].rolling(30, min_periods=30).mean()
    mrr = (g["close"] - mean_price).abs() / mean_price.replace(0, np.nan)

    s = mrr.astype(float)
    s.name = FEATURE_CODE
    return s
