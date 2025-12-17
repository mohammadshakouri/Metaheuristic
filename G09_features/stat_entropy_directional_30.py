FEATURE_CODE = "stat_entropy_directional_30"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Directional Entropy (30)
    Description:
        Measures the randomness of recent closing direction by computing Shannon entropy over a 30-bar window.
    Formula / method (brief, cite if needed):
        Let p_up and p_down be frequencies of positive and negative close-to-close changes within the window;
        entropy = -(p_up * log2(p_up) + p_down * log2(p_down)), ignoring zero terms.
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

    close = g["close"].astype(float)
    diff = close.diff()

    up = (diff > 0).astype(float)
    down = (diff < 0).astype(float)
    movers = (diff != 0).astype(float)

    window = 30
    up_count = up.rolling(window=window, min_periods=window).sum()
    down_count = down.rolling(window=window, min_periods=window).sum()
    move_count = movers.rolling(window=window, min_periods=window).sum()

    # Avoid division by zero by treating empty windows as NaN.
    denom = move_count.replace(0.0, np.nan)
    p_up = up_count / denom
    p_down = down_count / denom

    entropy_values = np.zeros(len(close), dtype=float)
    mask_up = p_up > 0
    mask_down = p_down > 0
    entropy_values[mask_up.fillna(False)] -= (p_up[mask_up] * np.log2(p_up[mask_up])).to_numpy()
    entropy_values[mask_down.fillna(False)] -= (p_down[mask_down] * np.log2(p_down[mask_down])).to_numpy()

    entropy = pd.Series(entropy_values, index=close.index, name=FEATURE_CODE)
    entropy = entropy.where(~denom.isna())
    return entropy.astype(float)
