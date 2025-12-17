FEATURE_CODE = "reg_persist_dir_10"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Persistent Direction Score (10)

    Description:
    Computes the fraction of the last 10 returns that were positive,
    scaled to [-1, 1]. Indicates recent directional bias.

    Formula:
    sign_t = 1 if ret>0, -1 if ret<0, 0 otherwise
    s_t = mean(sign_{t-9:t})

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    ret = g["close"].diff()
    sign = ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    s = sign.rolling(10, min_periods=10).mean()
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
