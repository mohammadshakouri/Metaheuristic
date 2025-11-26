FEATURE_CODE = "dyn_trend_angle_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trend Angle based on 30-period linear regression.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close = g["close"]
    angles = pd.Series(index=close.index, dtype=float)

    for i in range(29, len(close)):
        y = close[i-29:i+1]
        x = np.arange(30)
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        angle = np.arctan(m) * (180 / np.pi)
        angles.iat[i] = angle

    angles.name = FEATURE_CODE
    return angles
