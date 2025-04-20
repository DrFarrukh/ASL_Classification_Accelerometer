import numpy as np
import random
import os

# --- Augmentation functions ---
def add_noise(sample, noise_level=0.01):
    return sample + np.random.normal(0, noise_level, sample.shape)

def time_shift(sample, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(sample, shift, axis=1)  # Shift along time axis

def scale_signal(sample, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return sample * scale

def augment(sample):
    # Randomly apply one or more augmentations
    aug = sample.copy()
    if np.random.rand() < 0.5:
        aug = add_noise(aug)
    if np.random.rand() < 0.5:
        aug = time_shift(aug)
    if np.random.rand() < 0.5:
        aug = scale_signal(aug)
    return aug

# --- Main upsampling script ---
INPUT_NPZ = 'asl_all_samples.npz'
OUTPUT_NPZ = 'asl_upsampled_augmented.npz'

print(f"Loading {INPUT_NPZ} ...")
data = np.load(INPUT_NPZ, allow_pickle=True)
arrays = data['arrays']  # shape: (N, 5, 90, 6)
labels = data['labels']

# Organize samples by class
class_samples = {}
for arr, label in zip(arrays, labels):
    class_samples.setdefault(label, []).append(arr)

# Find max class size (should be 'Rest')
max_class_size = max(len(samples) for samples in class_samples.values())
print(f"Max class size: {max_class_size}")

new_arrays = []
new_labels = []
for label, samples in class_samples.items():
    n_existing = len(samples)
    n_needed = max_class_size - n_existing
    # Always include original samples
    new_arrays.extend(samples)
    new_labels.extend([label] * n_existing)
    # Oversample with augmentation
    for _ in range(n_needed):
        orig = random.choice(samples)
        aug = augment(orig)
        new_arrays.append(aug)
        new_labels.append(label)
    print(f"Class {label}: {n_existing} -> {max_class_size}")

new_arrays = np.stack(new_arrays)
new_labels = np.array(new_labels)

print(f"Saving upsampled/augmented dataset to {OUTPUT_NPZ} ...")
np.savez_compressed(OUTPUT_NPZ, arrays=new_arrays, labels=new_labels)
print("Done.")
