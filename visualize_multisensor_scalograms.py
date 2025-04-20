import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.colors import Normalize

# --- Load first sample ---
data = np.load('asl_all_samples.npz', allow_pickle=True)
arrays = data['arrays']  # shape: (N, 5, 90, 6)
labels = data['labels']
first_sample = arrays[0]  # shape: (5, 90, 6)

# --- Function to create RGB scalogram for one sensor ---
def sensor_scalogram(sensor_data, scales=np.arange(1, 31), wavelet='morl'):
    # sensor_data: shape (90, 6) for one sensor
    rgb = np.zeros((len(scales), sensor_data.shape[0], 3))
    color_indices = {'x': 0, 'y': 1, 'z': 2}
    for i, axis in enumerate(['x', 'y', 'z']):
        ch_idx = i  # 0: x, 1: y, 2: z
        sig = sensor_data[:, ch_idx]
        cwtmatr, _ = pywt.cwt(sig, scales, wavelet)
        norm = Normalize()(np.abs(cwtmatr))
        rgb[..., i] = norm
    # Normalize RGB to [0, 1]
    rgb = rgb / np.max(rgb)
    return rgb

# --- Create scalograms for all sensors ---
scalograms = []
for sensor_idx in range(first_sample.shape[0]):
    sensor_data = first_sample[sensor_idx]  # shape: (90, 6)
    rgb_scalogram = sensor_scalogram(sensor_data)
    scalograms.append(rgb_scalogram)

# --- Plot all scalograms side by side ---
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(scalograms[i], aspect='auto', origin='lower')
    ax.set_title(f'Sensor {i}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('multisensor_scalograms.png')
plt.show()
print('Saved multisensor scalogram visualization as multisensor_scalograms.png')
