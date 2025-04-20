import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pywt
from matplotlib.colors import Normalize
from PIL import Image
import matplotlib.pyplot as plt
import io

# ---- Config ----
MODEL_PATH = 'best_scalogram_cnn.pth'  # Update if your model file is named differently
IMG_SIZE = 128
SCALES = np.arange(1, 31)
WAVELET = 'morl'
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','Rest','S','T','U','V','W','X','Y','Z'
]

# ---- Model definition (should match your train_cnn_scalogram.py) ----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE//8) * (IMG_SIZE//8), 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---- Combined Scalogram Grid Generation ----
def rgb_scalogram_3axis(sig_x, sig_y, sig_z, scales=SCALES, wavelet=WAVELET):
    rgb = np.zeros((len(scales), sig_x.shape[0], 3))
    for i, sig in enumerate([sig_x, sig_y, sig_z]):
        cwtmatr, _ = pywt.cwt(sig, scales, wavelet)
        norm = Normalize()(np.abs(cwtmatr))
        rgb[..., i] = norm
    rgb = rgb / np.max(rgb)
    return rgb

def create_grid_scalogram(sample):
    # sample: (5, 90, 6)
    row_gyro, row_acc = [], []
    for sensor_idx in range(5):
        sensor_data = sample[sensor_idx]  # (90, 6)
        gyro_img = rgb_scalogram_3axis(sensor_data[:,0], sensor_data[:,1], sensor_data[:,2])
        acc_img = rgb_scalogram_3axis(sensor_data[:,3], sensor_data[:,4], sensor_data[:,5])
        row_gyro.append(gyro_img)
        row_acc.append(acc_img)
    grid_row1 = np.concatenate(row_gyro, axis=1)
    grid_row2 = np.concatenate(row_acc, axis=1)
    grid_img = np.concatenate([grid_row1, grid_row2], axis=0)  # (2*scales, 5*90, 3)
    # Resize to (IMG_SIZE, IMG_SIZE)
    img = (grid_img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return pil_img

# ---- Streamlit App ----
st.title('ASL Combined ScalogramGrid Classifier')
st.write('Upload a raw sensor signal file (.npz or .npy, shape (5, 90, 6)). The app will generate a combined grid scalogram and classify the gesture.')

uploaded_file = st.file_uploader('Upload raw signal file', type=['npz', 'npy'])

if uploaded_file:
    # Load sample
    if uploaded_file.name.endswith('.npz'):
        data = np.load(uploaded_file, allow_pickle=True)
        if 'arrays' in data:
            sample = data['arrays'][0]  # Use first sample
        else:
            st.error('No "arrays" key found in uploaded .npz file.')
            st.stop()
    else:
        sample = np.load(uploaded_file, allow_pickle=True)
    # Check shape
    if sample.shape != (5, 90, 6):
        st.error(f'Expected shape (5, 90, 6), got {sample.shape}')
        st.stop()
    # Generate grid scalogram
    scalogram_img = create_grid_scalogram(sample)
    st.image(scalogram_img, caption='Generated Combined Scalogram Grid', use_column_width=True)
    # Prepare for model
    img_arr = np.array(scalogram_img).astype(np.float32) / 255.0
    img_arr = np.transpose(img_arr, (2, 0, 1))  # (3, H, W)
    img_tensor = torch.tensor(img_arr).unsqueeze(0)
    # Load model
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = CLASS_NAMES[pred_idx]
    st.markdown(f'### Prediction: **{pred_label}**')
    st.bar_chart(probs)
    st.write('Top-5 Probabilities:')
    top5_idx = probs.argsort()[-5:][::-1]
    for i in top5_idx:
        st.write(f'{CLASS_NAMES[i]}: {probs[i]:.3f}')
else:
    st.info('Please upload a raw signal file to begin.')
