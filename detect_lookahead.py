import os
import sys
import glob
import importlib.util
import pandas as pd
import numpy as np

# Set up paths
PHASE1_FEATURES_DIR = './G09_features'
TEST_DATA_PATH = 'eth_5m_test.csv'


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_lookahead(feature_file, df):
    try:
        module = load_module_from_path(feature_file)
        if not hasattr(module, 'compute_feature'):
            return None  # Not a feature file or different structure

        # Run on original data
        # We use a copy to ensure the function doesn't modify the input in place
        s_orig = module.compute_feature(df.copy())

        # Create perturbed data - modify the last row
        df_perturbed = df.copy()

        # Perturb close, high, low, volume to catch dependencies on any of them
        # We modify the last row values significantly
        df_perturbed.iloc[-1, df_perturbed.columns.get_loc('close')] *= 1.05
        df_perturbed.iloc[-1, df_perturbed.columns.get_loc('high')] *= 1.05
        df_perturbed.iloc[-1, df_perturbed.columns.get_loc('low')] *= 0.95
        df_perturbed.iloc[-1, df_perturbed.columns.get_loc('volume')] *= 1.5

        s_perturbed = module.compute_feature(df_perturbed)

        # Compare all values EXCEPT the last one
        # If s_orig[i] != s_perturbed[i] for any i < last_index, then lookahead exists

        # Align series just in case (drop the last element)
        s_orig_check = s_orig.iloc[:-1]
        s_perturbed_check = s_perturbed.iloc[:-1]

        # Check for differences

        # Mask where both are NaN (safe)
        mask_both_nan = s_orig_check.isna() & s_perturbed_check.isna()

        # Mask where one is NaN and other is not -> definitely diff
        mask_one_nan = s_orig_check.isna() ^ s_perturbed_check.isna()

        # Mask where values differ significantly
        # We use a small epsilon for float comparison
        diff = np.abs(s_orig_check - s_perturbed_check)
        mask_diff_val = (diff > 1e-9) & (~mask_both_nan)

        has_lookahead = mask_one_nan.any() or mask_diff_val.any()

        if has_lookahead:
            # Find first index of failure
            if mask_one_nan.any():
                first_fail_idx = np.where(mask_one_nan)[0][0]
                first_fail_ts = s_orig_check.index[first_fail_idx]
            else:
                first_fail_idx = np.where(mask_diff_val)[0][0]
                first_fail_ts = s_orig_check.index[first_fail_idx]

            return f"FAIL: Lookahead detected at index {first_fail_idx} ({first_fail_ts})"

        return "PASS"

    except Exception as e:
        return f"ERROR: {str(e)}"


def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return

    print(f"Loading test data from {TEST_DATA_PATH}...")
    df = pd.read_csv(TEST_DATA_PATH)

    # Ensure timestamp is datetime if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Use a subset for speed, but large enough to have valid feature values
    # 1000 rows should be enough for most indicators to warm up
    df = df.head(1000).copy()

    print(f"Scanning features in {PHASE1_FEATURES_DIR}...")
    feature_files = glob.glob(os.path.join(
        PHASE1_FEATURES_DIR, '*.py'))
    feature_files.sort()

    results = []

    for f_path in feature_files:
        f_name = os.path.basename(f_path)
        print(f"Checking {f_name}...", end=' ', flush=True)
        result = check_lookahead(f_path, df)
        print(result)
        if result and "FAIL" in result:
            results.append((f_name, result))

    print("\nSummary of Failures:")
    if not results:
        print("No lookahead bias detected.")
    else:
        for name, res in results:
            print(f"{name}: {res}")


if __name__ == "__main__":
    main()
