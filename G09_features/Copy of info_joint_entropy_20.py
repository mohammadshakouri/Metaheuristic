FEATURE_CODE = "info_joint_entropy_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Information Joint Entropy of Price and Volume over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_returns = g["close"].pct_change()
    vol_changes = g["volume"].pct_change()

    joint_entropy = pd.Series(index=g.index, dtype=float)

    for i in range(19, len(g)):
        pr = price_returns[i-19:i+1]
        vc = vol_changes[i-19:i+1]
        joint_dist = pd.crosstab(pr, vc, normalize=True)
        joint_entropy.iat[i] = -np.sum(joint_dist.values * np.log(joint_dist.values + 1e-10))

    joint_entropy.name = FEATURE_CODE
    return joint_entropy
