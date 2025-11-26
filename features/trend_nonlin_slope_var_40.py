FEATURE_CODE = "trend_nonlin_slope_var_40"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trend Non-Linear Slope Variance over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close = g["close"]
    slope_var = pd.Series(index=close.index, dtype=float)

    for i in range(39, len(close)):
        y = close[i-39:i+1]
        x = np.arange(40)
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        slope_var.iat[i] = np.var(residuals)

    slope_var.name = FEATURE_CODE
    return slope_var
