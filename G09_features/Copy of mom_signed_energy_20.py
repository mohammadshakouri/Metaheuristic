FEATURE_CODE = "mom_signed_energy_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Signed Energy over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    signed_energy = delta.rolling(20, min_periods=20).apply(
        lambda x: np.sum(x**2) * np.sign(x[-1])
    )

    s = signed_energy.astype(float)
    s.name = FEATURE_CODE
    return s
