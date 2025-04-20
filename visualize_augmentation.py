import numpy as np
import matplotlib.pyplot as plt
import random

# --- Augmentation functions (same as in upsampling script) ---
def add_noise(sample, noise_level=0.01):
    return sample + np.random.normal(0, noise_level, sample.shape)

def time_shift(sample, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(sample, shift, axis=1)  # Shift along time axis

def scale_signal(sample, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return sample * scale

def augment(sample):
    aug = sample.copy()
    if np.random.rand() < 0.5:
        aug = add_noise(aug)
    if np.random.rand() < 0.5:
        aug = time_shift(aug)
    if np.random.rand() < 0.5:
        aug = scale_signal(aug)
    return aug

# --- Load a few samples ---
INPUT_NPZ = 'asl_all_samples.npz'
data = np.load(INPUT_NPZ, allow_pickle=True)
arrays = data['arrays']  # shape: (N, 5, 90, 6)
labels = data['labels']
feature_names = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']

# Pick a random sample from a non-Rest class
idx = next(i for i, l in enumerate(labels) if l != 'Rest')
sample = arrays[idx]  # shape: (5, 90, 6)

# Apply augmentations
noise_sample = add_noise(sample)
shift_sample = time_shift(sample)
scale_sample = scale_signal(sample)
combo_sample = augment(sample)

# Plot original and augmented signals for the first sensor (all features)
def plot_signals(orig, aug, title, fname):
    plt.figure(figsize=(10, 6))
    for i in range(orig.shape[-1]):
        plt.plot(orig[0,:,i], label=f'Orig-{feature_names[i]}', linestyle='-')
        plt.plot(aug[0,:,i], label=f'Aug-{feature_names[i]}', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

plot_signals(sample, noise_sample, 'Add Noise Augmentation (Sensor 0)', 'aug_noise.png')
plot_signals(sample, shift_sample, 'Time Shift Augmentation (Sensor 0)', 'aug_shift.png')
plot_signals(sample, scale_sample, 'Scaling Augmentation (Sensor 0)', 'aug_scale.png')
plot_signals(sample, combo_sample, 'Random Combo Augmentation (Sensor 0)', 'aug_combo.png')

print('Augmentation visualizations saved: aug_noise.png, aug_shift.png, aug_scale.png, aug_combo.png')
