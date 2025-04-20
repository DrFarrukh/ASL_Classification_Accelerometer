import os
import random
import pandas as pd
import numpy as np
from typing import Dict

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
CLASS_NAMES = [
    c for c in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, c)) and not c.startswith('.')
]

FEATURE_COLUMNS = [
    'gyro_x', 'gyro_y', 'gyro_z',
    'acc_x', 'acc_y', 'acc_z'
]

def load_all_samples_per_class(dataset_dir: str = DATASET_DIR) -> Dict[str, list]:
    """
    Loads all samples (Excel files) from each class folder (A-Z, Rest, etc.).
    For each class, returns a list of 3D numpy arrays (sensors, time_steps, features).
    {class_name: [sample_array, ...]}
    """
    class_samples = {}
    for class_name in CLASS_NAMES:
        class_path = os.path.join(dataset_dir, class_name)
        xlsx_files = [f for f in os.listdir(class_path) if f.endswith('.xlsx')]
        if not xlsx_files:
            continue
        sample_arrays = []
        for sample_file in xlsx_files:
            sample_path = os.path.join(class_path, sample_file)
            xl = pd.ExcelFile(sample_path)
            sensor_arrays = []
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                feature_df = df.iloc[:, 2:8]  # columns 3-8
                feature_df.columns = FEATURE_COLUMNS
                sensor_arrays.append(feature_df.values)  # shape: (time_steps, features)
            min_timesteps = min(arr.shape[0] for arr in sensor_arrays)
            sensor_arrays = [arr[:min_timesteps, :] for arr in sensor_arrays]
            sample_array = np.stack(sensor_arrays, axis=0)  # (sensors, time_steps, features)
            sample_arrays.append(sample_array)
        class_samples[class_name] = sample_arrays
    return class_samples

if __name__ == "__main__":
    all_samples = load_all_samples_per_class()
    # Save as a compressed npz file. Since lists of arrays can't be saved directly, flatten to arrays and save class labels.
    npz_dict = {}
    labels = []
    arrays = []
    for class_name, arr_list in all_samples.items():
        for arr in arr_list:
            arrays.append(arr)
            labels.append(class_name)
    arrays = np.array(arrays, dtype=object)  # Save as object array for variable shapes
    labels = np.array(labels)
    np.savez_compressed('asl_all_samples.npz', arrays=arrays, labels=labels)
    print(f"Saved all samples ({len(arrays)}) to 'asl_all_samples.npz'.")
