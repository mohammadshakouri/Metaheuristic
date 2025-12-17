FEATURE_CODE = "pat_hl_bias_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    High-Low Bias Score (20)

    Description:
    Measures whether closes are clustering near highs or lows of bars
    in the last 20 periods. Useful for identifying bullish or bearish bias.

    Formula:
    bias_t = mean( (close - low) / (high - low) ) over 20 bars

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    rng = (g["close"] - g["low"]) / (g["high"] - g["low"]).replace(0, np.nan)
    s = rng.rolling(20, min_periods=20).mean()

    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
