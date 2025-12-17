FEATURE_CODE = "trend_hma_slope_21"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Hull Moving Average Slope (21)

    Description:
    Computes the slope of the 21-period Hull Moving Average (HMA),
    a fast and smooth trend indicator. The slope approximates the
    short-term trend acceleration.

    Formula:
    WMA_n(x) = weighted moving average of window n
    HMA_21 = WMA(2*WMA(close,10) - WMA(close,21), sqrt(21))
    slope_t = HMA_t - HMA_{t-1}

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"]

    def wma(x, w):
        return (x * w).sum() / w.sum()

    w21 = np.arange(1, 22)
    w10 = np.arange(1, 11)
    wS = np.arange(1, int(np.sqrt(21)) + 1)

    wma21 = close.rolling(21).apply(lambda x: wma(x, w21), raw=True)
    wma10 = close.rolling(10).apply(lambda x: wma(x, w10), raw=True)

    fast = 2 * wma10 - wma21
    hma = fast.rolling(len(wS)).apply(lambda x: wma(x, wS), raw=True)

    slope = hma.diff()
    slope = slope.astype(float)
    slope.name = FEATURE_CODE
    return slope
