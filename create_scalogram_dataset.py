import numpy as np
import os
import pywt
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load balanced dataset
npz = np.load('asl_upsampled_augmented.npz', allow_pickle=True)
arrays = npz['arrays']
labels = npz['labels']
feature_names = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']

# Directory structure
base_dir = 'Scalograms'
acc_dir = os.path.join(base_dir, 'acc')
gyro_dir = os.path.join(base_dir, 'gyro')
os.makedirs(acc_dir, exist_ok=True)
os.makedirs(gyro_dir, exist_ok=True)

# Parameters for CWT
wavelet = 'morl'
scales = np.arange(1, 32)

# Helper to get RGB scalogram for 5 sensors, for gyro or acc
def rgb_scalogram(sample, feat_indices):
    rgb_sum = None
    for sensor in range(sample.shape[0]):
        rgb = np.zeros((len(scales), sample.shape[1], 3))
        for j, (fi, color) in enumerate(zip(feat_indices, [(1,0,0),(0,1,0),(0,0,1)])):
            sig = sample[sensor,:,fi]
            cwtmatr, _ = pywt.cwt(sig, scales, wavelet)
            norm = Normalize()(np.abs(cwtmatr))
            for k in range(3):
                rgb[:,:,k] += norm * color[k]
        if rgb_sum is None:
            rgb_sum = rgb
        else:
            rgb_sum += rgb
    rgb_sum = rgb_sum / np.max(rgb_sum)
    return rgb_sum

# Make class folders
def ensure_class_folder(parent_dir, class_name):
    class_dir = os.path.join(parent_dir, str(class_name))
    os.makedirs(class_dir, exist_ok=True)
    return class_dir

logging.info('Starting scalogram dataset creation...')
class_counts = {}
for idx, (arr, label) in enumerate(zip(arrays, labels)):
    # arr: (sensors, time_steps, features)
    # Save acc
    acc_rgb = rgb_scalogram(arr, [3,4,5])
    acc_class_dir = ensure_class_folder(acc_dir, label)
    acc_path = os.path.join(acc_class_dir, f'{idx}.png')
    fig_acc, ax_acc = plt.subplots(figsize=(6, 6), dpi=150)
    ax_acc.imshow(acc_rgb, aspect='auto', origin='lower')
    ax_acc.axis('off')
    fig_acc.tight_layout(pad=0)
    fig_acc.savefig(acc_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_acc)
    # Save gyro
    gyro_rgb = rgb_scalogram(arr, [0,1,2])
    gyro_class_dir = ensure_class_folder(gyro_dir, label)
    gyro_path = os.path.join(gyro_class_dir, f'{idx}.png')
    fig_gyro, ax_gyro = plt.subplots(figsize=(6, 6), dpi=150)
    ax_gyro.imshow(gyro_rgb, aspect='auto', origin='lower')
    ax_gyro.axis('off')
    fig_gyro.tight_layout(pad=0)
    fig_gyro.savefig(gyro_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_gyro)
    # Logging
    class_counts[label] = class_counts.get(label, 0) + 1
    if idx % 100 == 0:
        logging.info(f'Saved sample {idx}: class={label}')
logging.info('Done! Saved scalograms for all samples.')
logging.info(f'Class sample counts: {class_counts}')
