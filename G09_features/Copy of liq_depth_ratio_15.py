FEATURE_CODE = "liq_depth_ratio_15"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Depth Ratio over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    avg_volume = g["volume"].rolling(15, min_periods=15).mean()
    price_range = g["high"] - g["low"]
    ldr = avg_volume / price_range.replace(0, np.nan)

    s = ldr.astype(float)
    s.name = FEATURE_CODE
    return s
