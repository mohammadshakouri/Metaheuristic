FEATURE_CODE = "log_return_low"
import numpy as np, pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Log Return of Low Prices
    Description:
    This feature measures the day-to-day percentage change in the *Low* price on a logarithmic scale.
    It captures short-term volatility or momentum in the lower price bounds of each candle.
    Log returns are commonly used in finance because they are time-additive and stabilize variance.

    Formula / Method:
    log_return_low_t = ln( low_t / low_(t-1) )

    Input:
    df: DataFrame with DatetimeIndex (ascending), columns:
    open, high, low, close, volume (case-insensitive)

    Output:
    pd.Series (float), same index as df.index, length == len(df),
    name == FEATURE_CODE. The first value will be NaN because there is no previous data point.

    Constraints:
    - No look-ahead bias (only uses current and previous high).
    - Fully vectorized using pandas / numpy (no loops).
    - Allowed to have NaN at the start (due to shift).
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    s = np.log(df["low"] / df["low"].shift(1))
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
