import numpy as np
import os
import pywt
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import logging

# --- Config ---
INPUT_NPZ = 'asl_upsampled_augmented.npz'
OUTPUT_DIR = 'CombinedScalograms'
SCALES = np.arange(1, 31)
WAVELET = 'morl'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- Function to create RGB scalogram for one sensor ---
def sensor_scalogram(sensor_data, scales=SCALES, wavelet=WAVELET):
    # sensor_data: shape (90, 6) for one sensor
    rgb = np.zeros((len(scales), sensor_data.shape[0], 3))
    for i, axis in enumerate(['x', 'y', 'z']):
        sig = sensor_data[:, i]  # 0: x, 1: y, 2: z
        cwtmatr, _ = pywt.cwt(sig, scales, wavelet)
        norm = Normalize()(np.abs(cwtmatr))
        rgb[..., i] = norm
    rgb = rgb / np.max(rgb)
    return rgb

# --- Load dataset ---
data = np.load(INPUT_NPZ, allow_pickle=True)
arrays = data['arrays']  # shape: (N, 5, 90, 6)
labels = data['labels']

os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, (sample, label) in enumerate(zip(arrays, labels)):
    # For each sensor, create RGB scalogram
    scalograms = []
    for sensor_idx in range(sample.shape[0]):
        sensor_data = sample[sensor_idx]  # shape: (90, 6)
        rgb_scalogram = sensor_scalogram(sensor_data)
        scalograms.append(rgb_scalogram)
    # Stack all 5 sensor scalograms horizontally (axis=1)
    combined_img = np.concatenate(scalograms, axis=1)  # (scales, 5*90, 3)
    # Save image
    class_dir = os.path.join(OUTPUT_DIR, str(label))
    os.makedirs(class_dir, exist_ok=True)
    out_path = os.path.join(class_dir, f'{idx}.png')
    plt.imsave(out_path, combined_img)
    if idx % 100 == 0:
        logging.info(f'Saved combined scalogram for sample {idx}: class={label}')
logging.info('Done! Saved combined scalograms for all samples.')
