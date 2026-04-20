import numpy as np
import pandas as pd
from pathlib import Path
import sys

def convert_csv_to_npy(csv_path, npy_path):
    print(f"Loading {csv_path}...")
    # Load without header as it seems it doesn't have one
    data = pd.read_csv(csv_path, header=None).values.astype(np.float64)
    print(f"Shape: {data.shape}")
    np.save(npy_path, data)
    print(f"Saved to {npy_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default for testing
        root = Path(r"C:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization-by-ResMLP128\TLIO_Oxford_Dataset")
        seq = "oxford_handheld_1"
        csv_file = root / seq / "imu_samples_0.csv"
        npy_file = root / seq / "imu0_resampled.npy"
        convert_csv_to_npy(csv_file, npy_file)
    else:
        convert_csv_to_npy(sys.argv[1], sys.argv[2])
