FEATURE_CODE = "vol_true_range_zscore_20"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    ATR Z-Score (20)
    Description:
        Standardizes the current 14-period Average True Range (ATR) against its own 20-bar history to highlight volatility shocks.
    Formula / method (brief, cite if needed):
        Compute ATR(14); z-score = (ATR(14) - mean(ATR(14),20)) / std(ATR(14),20); zero standard deviations yield NaN.
    Input:
        df: DataFrame with DatetimeIndex (ascending), columns:
        open, high, low, close, volume (case-insensitive)
    Output:
        pd.Series (float preferred), same index as df.index, length == len(df),
        name == FEATURE_CODE. Initial NaNs from rolling windows are OK.
    Constraints:
        - No look-ahead (use only current and past data at each row).
        - Vectorized (avoid Python loops when possible).
        - Use only numpy and pandas.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    atr_14 = true_range.rolling(window=14, min_periods=14).mean()
    atr_mean = atr_14.rolling(window=20, min_periods=20).mean()
    atr_std = atr_14.rolling(window=20, min_periods=20).std()

    zscore = (atr_14 - atr_mean) / atr_std.replace(0.0, np.nan)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    zscore = zscore.astype(float)
    zscore.name = FEATURE_CODE
    return zscore
