import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
import os

# Create output folder for plots
plot_dir = 'eda_plots'
os.makedirs(plot_dir, exist_ok=True)

# Load balanced dataset
npz = np.load('asl_balanced_samples.npz', allow_pickle=True)
arrays = npz['arrays']
labels = npz['labels']
feature_names = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']

print(f"Total samples: {len(arrays)}")
print(f"Classes: {sorted(set(labels))}")
print("Class counts:", Counter(labels))

# --- Scalogram RGB plots for first sample of each class ---
import pywt
from matplotlib.colors import Normalize

# Get first sample per class
class_first_sample = {}
for arr, label in zip(arrays, labels):
    if label not in class_first_sample:
        class_first_sample[label] = arr  # arr: (sensors, time_steps, features)

classes = sorted(class_first_sample.keys())
num_classes = len(classes)

# Parameters for CWT
wavelet = 'morl'
scales = np.arange(1, 32)

# Helper to get RGB scalogram for 5 sensors, for gyro or acc
# gyro_indices = [0,1,2], acc_indices = [3,4,5]
def rgb_scalogram(sample, feat_indices):
    # sample: (sensors, time_steps, features)
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
    # Normalize after summing sensors
    rgb_sum = rgb_sum / np.max(rgb_sum)
    return rgb_sum

# Plot all classes: gyro and acc in a grid
import math
n_cols = 6
n_rows = math.ceil(num_classes / n_cols)
fig_gyro, axs_gyro = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
fig_acc, axs_acc = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
axs_gyro = axs_gyro.flatten()
axs_acc = axs_acc.flatten()
for idx, cls in enumerate(classes):
    sample = class_first_sample[cls]
    gyro_rgb = rgb_scalogram(sample, [0,1,2])
    acc_rgb = rgb_scalogram(sample, [3,4,5])
    axs_gyro[idx].imshow(gyro_rgb, aspect='auto', origin='lower')
    axs_gyro[idx].set_title(f'{cls}')
    axs_gyro[idx].axis('off')
    axs_acc[idx].imshow(acc_rgb, aspect='auto', origin='lower')
    axs_acc[idx].set_title(f'{cls}')
    axs_acc[idx].axis('off')
# Hide unused axes
for ax in axs_gyro[num_classes:]:
    ax.axis('off')
for ax in axs_acc[num_classes:]:
    ax.axis('off')
fig_gyro.suptitle('Gyro RGB Scalograms (First Sample per Class)', y=1.02)
fig_acc.suptitle('Acc RGB Scalograms (First Sample per Class)', y=1.02)
fig_gyro.tight_layout(rect=[0, 0, 1, 0.97])
fig_acc.tight_layout(rect=[0, 0, 1, 0.97])
fig_gyro.savefig(os.path.join(plot_dir, 'gyro_scalograms_all_classes.png'))
fig_acc.savefig(os.path.join(plot_dir, 'acc_scalograms_all_classes.png'))
plt.close(fig_gyro)
plt.close(fig_acc)

# Flatten all data for EDA
all_data = []
all_labels = []
for arr, label in zip(arrays, labels):
    # arr: (sensors, time_steps, features)
    all_data.append(arr.reshape(-1, arr.shape[2]))
    all_labels.extend([label] * (arr.shape[0] * arr.shape[1]))
all_data = np.concatenate(all_data, axis=0)

# 1. Per-feature statistics
print("\nPer-feature statistics:")
for i, name in enumerate(feature_names):
    col = all_data[:, i]
    print(f"  {name:8s}: mean={np.mean(col):8.2f}, std={np.std(col):8.2f}, min={np.min(col):8.2f}, max={np.max(col):8.2f}")

# 2. Feature histograms
plt.figure(figsize=(14, 8))
for i, name in enumerate(feature_names):
    plt.subplot(2, 3, i+1)
    sns.histplot(all_data[:, i], bins=50, kde=True)
    plt.title(name)
plt.tight_layout()
plt.suptitle('Feature Distributions (All Classes)', y=1.02)
plt.savefig(os.path.join(plot_dir, 'feature_distributions.png'))
plt.close()

# 3. Feature boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=[all_data[:, i] for i in range(len(feature_names))])
plt.xticks(range(6), feature_names)
plt.title('Feature Boxplots')
plt.savefig(os.path.join(plot_dir, 'feature_boxplots.png'))
plt.close()

# 4. Correlation matrix (using pandas for robustness)
import pandas as pd
all_data_df = pd.DataFrame(all_data, columns=feature_names)
corr = all_data_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names)
plt.title('Feature Correlation Matrix')
plt.savefig(os.path.join(plot_dir, 'feature_correlation_matrix.png'))
plt.close()

# 5. PCA for visualization (optional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(all_data)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.factorize(all_labels)[0], cmap='tab20', s=1, alpha=0.3)
plt.title('PCA Projection (All Data Points)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(os.path.join(plot_dir, 'pca_projection.png'))
plt.close()

print("EDA complete. You can add more plots or per-class analyses as needed.")
