FEATURE_CODE = "log_return_open"
import numpy as np, pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Log Return of Open Prices
    Description:
    This feature measures the day-to-day percentage change in the *open* price on a natural logarithmic scale.
    It helps capture how the market opens relative to the previous day, which can reflect overnight sentiment,
    news impact, and gap behavior. Log returns are commonly used because they are time-additive and reduce
    scale-related distortions.

    Formula / Method:
    log_return_open_t = ln( open_t / open_(t-1) )

    Input:
    df: DataFrame with DatetimeIndex (ascending), columns:
    open, high, low, close, volume (case-insensitive)

    Output:
    pd.Series (float), same index as df.index, length == len(df),
    name == FEATURE_CODE. The first element will be NaN due to the required past value.

    Constraints:
    - No look-ahead bias (depends only on current and previous open values).
    - Fully vectorized using pandas/numpy (no Python loops).
    - NaN at the beginning is expected and acceptable.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    s = np.log(df["open"] / df["open"].shift(1))
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
