FEATURE_CODE = "info_directional_variability_25"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Information Directional Variability over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    directional_changes = (delta > 0).astype(int) - (delta < 0).astype(int)

    variability = directional_changes.rolling(25, min_periods=25).apply(
        lambda x: -np.sum((x.value_counts(normalize=True) * np.log(x.value_counts(normalize=True) + 1e-10)))
    )

    s = variability.astype(float)
    s.name = FEATURE_CODE
    return s
