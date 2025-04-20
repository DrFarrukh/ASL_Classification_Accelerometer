import numpy as np

# Load the compressed npz file containing random samples
npz = np.load('asl_random_samples.npz')

print("Loaded classes:", list(npz.keys()))

import matplotlib.pyplot as plt

# Plot a single class sample (e.g., 'A')
class_to_plot = 'A'
arr = npz[class_to_plot]  # shape: (sensors, time_steps, features)
feature_names = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
num_sensors = arr.shape[0]
num_features = arr.shape[2]
time_steps = arr.shape[1]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# 3D surface plots for all 5 sensors
fig = plt.figure(figsize=(22, 4))
for sensor_idx in range(num_sensors):
    sensor_data = arr[sensor_idx]  # shape: (time_steps, features)
    time_steps_arr = np.arange(sensor_data.shape[0])
    features_arr = np.arange(sensor_data.shape[1])
    T, F = np.meshgrid(time_steps_arr, features_arr)
    Z = sensor_data.T  # (features, time_steps)

    ax = fig.add_subplot(1, num_sensors, sensor_idx+1, projection='3d')
    surf = ax.plot_surface(T, F, Z, cmap=cm.viridis, edgecolor='none', alpha=0.8)
    ax.set_xlabel('Time')
    if sensor_idx == 0:
        ax.set_ylabel('Feature')
        ax.set_yticks(features_arr)
        ax.set_yticklabels(feature_names)
    else:
        ax.set_yticks([])
    ax.set_zlabel('Value')
    ax.set_title(f'Sensor {sensor_idx+1}')
    # Only add colorbar to the last subplot for clarity
    if sensor_idx == num_sensors-1:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label='Sensor value')
plt.suptitle(f'3D Surface Plots - Class {class_to_plot}, All Sensors', y=1.05, fontsize=16)
plt.tight_layout()
plt.show()
