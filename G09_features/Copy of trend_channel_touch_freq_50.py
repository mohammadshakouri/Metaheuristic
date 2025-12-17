FEATURE_CODE = "trend_channel_touch_freq_50"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trend Channel Touch Frequency over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high = g["high"]
    low = g["low"]
    close = g["close"]

    channel_touches = pd.Series(index=close.index, dtype=float)

    for i in range(49, len(close)):
        upper_channel = high[i-49:i+1].max()
        lower_channel = low[i-49:i+1].min()
        touches = ((close[i-49:i+1] >= upper_channel) | (close[i-49:i+1] <= lower_channel)).sum()
        channel_touches.iat[i] = touches / 50

    channel_touches.name = FEATURE_CODE
    return channel_touches
