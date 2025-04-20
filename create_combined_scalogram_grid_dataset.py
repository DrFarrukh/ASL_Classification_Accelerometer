import numpy as np
import os
import pywt
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import logging

# --- Config ---
INPUT_NPZ = 'asl_upsampled_augmented.npz'
OUTPUT_DIR = 'CombinedScalogramsGrid'
SCALES = np.arange(1, 31)
WAVELET = 'morl'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- Function to create RGB scalogram for 3-axis data ---
def rgb_scalogram_3axis(sig_x, sig_y, sig_z, scales=SCALES, wavelet=WAVELET):
    rgb = np.zeros((len(scales), sig_x.shape[0], 3))
    for i, sig in enumerate([sig_x, sig_y, sig_z]):
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
    grid_imgs = []  # Will be 2x5 grid: [gyro0-4, acc0-4]
    # First row: gyro (features 0,1,2)
    row_gyro = []
    for sensor_idx in range(5):
        sensor_data = sample[sensor_idx]  # (90, 6)
        rgb_img = rgb_scalogram_3axis(sensor_data[:,0], sensor_data[:,1], sensor_data[:,2])
        row_gyro.append(rgb_img)
    # Second row: acc (features 3,4,5)
    row_acc = []
    for sensor_idx in range(5):
        sensor_data = sample[sensor_idx]  # (90, 6)
        rgb_img = rgb_scalogram_3axis(sensor_data[:,3], sensor_data[:,4], sensor_data[:,5])
        row_acc.append(rgb_img)
    # Stack rows
    grid_row1 = np.concatenate(row_gyro, axis=1)
    grid_row2 = np.concatenate(row_acc, axis=1)
    grid_img = np.concatenate([grid_row1, grid_row2], axis=0)  # shape: (2*scales, 5*90, 3)
    # Save image
    class_dir = os.path.join(OUTPUT_DIR, str(label))
    os.makedirs(class_dir, exist_ok=True)
    out_path = os.path.join(class_dir, f'{idx}.png')
    plt.imsave(out_path, grid_img)
    if idx % 100 == 0:
        logging.info(f'Saved grid scalogram for sample {idx}: class={label}')
logging.info('Done! Saved grid scalograms for all samples.')
