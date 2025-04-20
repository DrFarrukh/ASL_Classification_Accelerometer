import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pywt
from matplotlib.colors import Normalize
from PIL import Image
import io
import os

# --- Model definition (must match train_cnn_scalogram.py) ---
class SimpleJetsonCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Utility: Generate RGB scalogram for 3-axis data ---
def rgb_scalogram_3axis(sig_x, sig_y, sig_z, scales=np.arange(1,31), wavelet='morl'):
    rgb = np.zeros((len(scales), sig_x.shape[0], 3))
    for i, sig in enumerate([sig_x, sig_y, sig_z]):
        cwtmatr, _ = pywt.cwt(sig, scales, wavelet)
        norm = Normalize()(np.abs(cwtmatr))
        rgb[..., i] = norm
    rgb = rgb / np.max(rgb)
    return rgb

# --- Preprocess uploaded xlsx file ---
def preprocess_xlsx(xlsx_file):
    # Expecting 5 sheets, each with at least 90 rows and at least 8 columns
    xls = pd.ExcelFile(xlsx_file)
    sheets = xls.sheet_names
    if len(sheets) != 5:
        raise ValueError(f"Expected 5 sheets (one per sensor), got {len(sheets)}")
    sensor_arrays = []
    for sheet in sheets:
        df = xls.parse(sheet, header=None)
        if df.shape[1] < 8:
            raise ValueError(f"Sheet '{sheet}' has only {df.shape[1]} columns, expected at least 8 (need columns 3-8)")
        feature_df = df.iloc[:, 2:8]  # columns 3-8
        sensor_arrays.append(feature_df.values)
    min_timesteps = min(arr.shape[0] for arr in sensor_arrays)
    if min_timesteps < 90:
        raise ValueError(f"One or more sheets have less than 90 rows (minimum found: {min_timesteps})")
    if min_timesteps > 90:
        st.warning(f"Some sheets have more than 90 rows; only the first 90 will be used.")
    # Trim all sensors to first 90 timesteps
    trimmed_arrays = [arr[:90, :] for arr in sensor_arrays]
    arr = np.stack(trimmed_arrays, axis=0)  # (5, 90, 6)
    return arr

# --- Generate 2x5 grid scalogram image from sample ---
def make_scalogram_grid(sample):
    SCALES = np.arange(1, 31)
    WAVELET = 'morl'
    row_gyro = []
    row_acc = []
    for sensor_idx in range(5):
        sensor_data = sample[sensor_idx]  # (90, 6)
        # Gyro (0,1,2)
        rgb_gyro = rgb_scalogram_3axis(sensor_data[:,0], sensor_data[:,1], sensor_data[:,2], SCALES, WAVELET)
        row_gyro.append(rgb_gyro)
        # Acc (3,4,5)
        rgb_acc = rgb_scalogram_3axis(sensor_data[:,3], sensor_data[:,4], sensor_data[:,5], SCALES, WAVELET)
        row_acc.append(rgb_acc)
    grid_row1 = np.concatenate(row_gyro, axis=1)
    grid_row2 = np.concatenate(row_acc, axis=1)
    grid_img = np.concatenate([grid_row1, grid_row2], axis=0)  # (2*scales, 5*90, 3)
    return grid_img

# --- Main Streamlit App ---
st.title('ASL Signal Classification (Scalogram Grid Inference)')
st.write('Upload a raw ASL signal .xlsx file (shape: 5 sensors × 90 time steps × 6 features)')

uploaded_file = st.file_uploader('Choose an XLSX file', type=['xlsx'])

if uploaded_file:
    try:
        sample = preprocess_xlsx(uploaded_file)
        st.success('File loaded and parsed!')
        # --- Plot raw signals for each sensor ---
        feature_names = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for sensor_idx in range(5):
            fig, ax = plt.subplots(figsize=(8, 3))
            for feat_idx, feat_name in enumerate(feature_names):
                ax.plot(sample[sensor_idx, :, feat_idx], label=feat_name, color=colors[feat_idx])
            ax.set_title(f'Sensor {sensor_idx+1} Raw Signals')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Value')
            ax.legend(loc='upper right')
            st.pyplot(fig)
        # --- Generate grid scalogram ---
        grid_img = make_scalogram_grid(sample)
        st.markdown('---')
        st.subheader('Scalogram Grid')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(grid_img, aspect='auto', origin='lower')
        ax.axis('off')
        st.pyplot(fig)
        # Save to buffer for model input
        img_pil = Image.fromarray((grid_img*255).astype(np.uint8))
        img_pil = img_pil.resize((128, 128))
        img_arr = np.array(img_pil).astype(np.float32) / 255.0
        # Normalize as in train_cnn_scalogram.py
        img_arr = (img_arr - 0.5) / 0.5
        img_arr = np.transpose(img_arr, (2, 0, 1))  # (3, H, W)
        img_tensor = torch.tensor(img_arr).unsqueeze(0)
        # Load model
        model = SimpleJetsonCNN(num_classes=27)
        model.load_state_dict(torch.load('best_scalogram_cnn.pth', map_location='cpu'))
        model.eval()
        # Load class names from directory (as in training)
        data_root = 'CombinedScalogramsGrid'
        class_names = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        # Predict
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
        st.markdown('---')
        st.subheader('Classification Result')
        st.write(f'**Predicted Class:** {class_names[pred_idx]}')
        st.bar_chart(probs)
    except Exception as e:
        st.error(f'Error: {e}')
