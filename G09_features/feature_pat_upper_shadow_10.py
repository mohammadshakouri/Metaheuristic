FEATURE_CODE = "pat_upper_shadow_10"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Upper Shadow Strength (10)

    Description:
    Measures the relative size of the upper candle wick across the last
    10 bars. Values close to 1 indicate repeated selling pressure near
    highs. Helps detect exhaustion, rejection, or topping conditions.

    Formula:
    upper_shadow = high - max(open, close)
    body_range = abs(close - open)
    strength = SMA(upper_shadow,10) / SMA(body_range,10)

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    upper = g["high"] - g[["open", "close"]].max(axis=1)
    body = (g["close"] - g["open"]).abs()

    avg_upper = upper.rolling(10, min_periods=10).mean()
    avg_body = body.rolling(10, min_periods=10).mean()

    s = avg_upper / avg_body.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
