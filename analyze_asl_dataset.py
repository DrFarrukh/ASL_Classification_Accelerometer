import numpy as np
from collections import Counter, defaultdict

# Load the dataset
npz = np.load('asl_all_samples.npz', allow_pickle=True)
arrays = npz['arrays']  # shape: (num_samples,), each is (sensors, time_steps, features)
labels = npz['labels']  # shape: (num_samples,)
feature_names = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']

print(f"Total samples: {len(arrays)}")
print(f"Classes: {set(labels)}")

# 1. Number of samples per class
class_counts = Counter(labels)
print("\nNumber of samples per class:")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls}: {count}")

# 2. Distribution of sample lengths (time steps)
sample_lengths = [arr.shape[1] for arr in arrays]
print(f"\nSample length (time steps): min={np.min(sample_lengths)}, max={np.max(sample_lengths)}, mean={np.mean(sample_lengths):.1f}")

# 3. Per-feature statistics (mean, std, min, max) across all sensors and samples
all_data = []
for arr in arrays:
    # arr: (sensors, time_steps, features), flatten sensors and time_steps
    all_data.append(arr.reshape(-1, arr.shape[2]))
all_data = np.concatenate(all_data, axis=0)  # shape: (total_points, features)

print("\nPer-feature statistics across all samples and sensors:")
for i, name in enumerate(feature_names):
    col = all_data[:, i]
    print(f"  {name:8s}: mean={np.mean(col):8.2f}, std={np.std(col):8.2f}, min={np.min(col):8.2f}, max={np.max(col):8.2f}")

# 4. Optional: class-wise statistics
class_stats = defaultdict(list)
for arr, cls in zip(arrays, labels):
    class_stats[cls].append(arr.reshape(-1, arr.shape[2]))

print("\nPer-class sample count and mean length:")
for cls, arrs in sorted(class_stats.items()):
    lengths = [a.shape[0] for a in arrs]
    print(f"  {cls:6s}: count={len(arrs):3d}, mean_length={np.mean(lengths):5.1f}")

# --- Downsample 'Rest' class to match other classes ---
import random
random.seed(42)

rest_indices = [i for i, label in enumerate(labels) if label == 'Rest']
other_indices = [i for i, label in enumerate(labels) if label != 'Rest']

# Number to match
num_per_class = min(Counter(labels).values())
rest_downsampled = random.sample(rest_indices, num_per_class)

balanced_indices = other_indices + rest_downsampled
balanced_arrays = arrays[balanced_indices]
balanced_labels = labels[balanced_indices]

print(f"\nAfter downsampling 'Rest':")
from collections import Counter
print("Class counts:", Counter(balanced_labels))

# Save balanced dataset
np.savez_compressed('asl_balanced_samples.npz', arrays=balanced_arrays, labels=balanced_labels)
print("Saved balanced dataset as 'asl_balanced_samples.npz'.")
