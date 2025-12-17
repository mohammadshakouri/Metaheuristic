FEATURE_CODE = "pat_lower_shadow_pct_15"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Lower Shadow Percentage (15)

    Description:
    Measures the fraction of the candle's total range attributed to the
    lower shadow. Useful for detecting buying pressure.

    Formula:
    lower_shadow = min(open,close) - low
    pct = SMA(lower_shadow,15) / SMA(range,15)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    lower = g[["open", "close"]].min(axis=1) - g["low"]
    rng = g["high"] - g["low"]

    avg_lower = lower.rolling(15, min_periods=15).mean()
    avg_rng = rng.rolling(15, min_periods=15).mean()

    s = avg_lower / avg_rng.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
