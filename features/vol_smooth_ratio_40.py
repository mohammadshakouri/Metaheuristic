FEATURE_CODE = "vol_smooth_ratio_40"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volatility Smooth Ratio over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    smooth_vol = volatility.rolling(40, min_periods=40).mean()
    vsr = smooth_vol / volatility.replace(0, np.nan)

    s = vsr.astype(float)
    s.name = FEATURE_CODE
    return s
