FEATURE_CODE = "mom_energy_osc_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Energy Oscillator over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    momentum = g["close"].diff(1)
    energy = momentum.rolling(20, min_periods=20).apply(lambda x: np.sum(x**2))

    meo = momentum / energy.replace(0, np.nan)

    s = meo.astype(float)
    s.name = FEATURE_CODE
    return s
