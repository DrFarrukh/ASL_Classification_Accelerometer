# ASL Sensor Data Classification Project

## Overview
This project provides a complete pipeline for classifying American Sign Language (ASL) gestures using wearable sensor data. The workflow includes data preprocessing, exploratory data analysis, dataset balancing (with augmentation), scalogram generation, and training lightweight CNN models on both raw signals and scalogram images. The models are optimized for efficient deployment on edge devices like the NVIDIA Jetson Nano.

## Directory Structure
```
ASL/
├── create_scalogram_dataset.py         # Generate scalogram images from raw signals
├── create_upsampled_augmented_dataset.py # Create balanced/augmented dataset (NPZ)
├── eda_asl_balanced.py                 # Exploratory data analysis
├── load_asl_dataset.py                 # Load and process raw ASL data
├── load_npz_samples.py                 # Visualize NPZ samples
├── train_cnn_scalogram.py              # Train CNN on scalogram images
├── train_raw_signal_model.py           # Train 3D CNN on raw signals
├── Scalograms/                         # Generated scalogram images (RGB)
├── dataset_description.md              # Dataset structure and details
├── asl_balanced_samples.npz            # Balanced dataset (before upsampling)
├── asl_all_samples.npz                 # All available samples
├── asl_upsampled_augmented.npz         # Balanced and augmented dataset (for training)
├── .gitignore                          # Ignore rules for dataset, models, etc.
└── README.md                           # Project documentation
```

## Workflow

### 1. Data Preparation
- **Raw Data:** Sensor data for ASL gestures (accelerometer, gyroscope, etc.)
- **Balancing:**
  - `asl_balanced_samples.npz`: Downsampled to ensure equal samples per class (minority class count).
  - `asl_upsampled_augmented.npz`: All classes upsampled with augmentation to match the largest class ("Rest").
- **Augmentation:**
  - Noise addition, time shifting, and scaling are used to generate synthetic samples for minority classes.

### 2. Exploratory Data Analysis (EDA)
- Run `eda_asl_balanced.py` to visualize feature distributions, correlations, and class balance.
- Outputs: Histograms, boxplots, correlation matrices.

### 3. Scalogram Generation
- **Original:** Use `create_scalogram_dataset.py` to convert raw sensor signals into RGB scalogram images for each class, organized by class and sensor type (accelerometer/gyroscope).
- **Combined Grid Version:**
  - Use `create_combined_scalogram_grid_dataset.py` to generate a new dataset where, for each sample:
    - Each of the 5 sensors produces two RGB scalograms: one for gyro (x→R, y→G, z→B), one for acc (x→R, y→G, z→B).
    - These 10 scalograms are arranged in a 2x5 grid (top row: gyro, bottom row: acc) and saved as a single image per sample in `CombinedScalogramsGrid/`.
  - This approach provides the model with more spatial context and leverages all sensor axes in a visually intuitive way.

### 4. Model Training
- **CNN on Combined Scalogram Grids:**
  - `train_cnn_scalogram.py`: Trains a lightweight CNN on the new grid-style scalogram images from `CombinedScalogramsGrid/`.
  - Outputs: Model weights (`.pth`), training/validation curves, confusion matrix, classification report.
  - Achieved ~93% validation accuracy, with most classes above 0.9 F1-score.
  - The 'Rest' class remains challenging (f1 ≈ 0.47), which is common in gesture datasets.
- **3D CNN on Raw Signals:**
  - `train_raw_signal_model.py`: Trains a 3D CNN on the upsampled/augmented raw signal dataset.
  - Outputs: Model weights, training curves, confusion matrix, classification report.
  - Early stopping and class balancing supported.

### 5. Evaluation & Diagnostics
- Both training scripts save:
  - Training/validation loss and accuracy curves (`*_training_curves.png`)
  - Confusion matrix (`*_confusion_matrix.png`)
  - Classification report (`*_classification_report.txt`)
- **Sample Results (CombinedScalogramGrid):**
  - Validation Accuracy: 92.95% (best: 97.37%)
  - Macro F1-score: 0.93
  - Most classes F1 > 0.9; 'Rest' class F1 ≈ 0.47
  - See logs and output files for detailed per-class results.

### 6. Deployment
- Models are designed to be lightweight for deployment on edge devices (e.g., Jetson Nano).
- (ONNX export can be re-enabled if TensorRT optimization is needed.)

## Requirements
- Python 3.8+
- Key packages: `numpy`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `pywt`, `PIL`
- (For Jetson Nano: install PyTorch and ONNX as per NVIDIA's official guides)

## Usage
1. **Prepare Dataset:**
    - Place your raw data in the expected format (see `dataset_description.md`).
    - Run `create_upsampled_augmented_dataset.py` to generate a balanced, augmented dataset.
2. **Generate Scalograms:**
    - Run `create_scalogram_dataset.py` to create scalogram images.
3. **Train Models:**
    - For scalograms: `python train_cnn_scalogram.py`
    - For raw signals: `python train_raw_signal_model.py`
4. **Review Outputs:**
    - Check PNG and TXT files for model performance and diagnostics.
    - Best model weights are saved as `.pth` files.

## Customization
- **Augmentation:** Edit `create_upsampled_augmented_dataset.py` to adjust noise, shift, or scaling parameters.
- **Model Architectures:** Modify the CNNs in the training scripts for experimentation.
- **Grid Scalogram Structure:** Tune the arrangement or type of scalogram (e.g., try 1x10 or 5x2 layouts, or add more axes).
- **Logging:** Logging levels and formats can be changed in each script.

## Notes
- All large/binary files (`*.pth`, `*.npz`, images) are excluded from version control via `.gitignore`.
- For ONNX/TensorRT export, see commented code in training scripts.
- For best results, use the upsampled/augmented dataset and the grid-style scalogram images.

## Streamlit Real-Time Inference App

A user-friendly Streamlit app is included for real-time ASL gesture inference using the trained CNN model on grid-style scalograms.

### Features
- **Upload a raw `.xlsx` ASL signal file** (5 sheets, each with at least 90 rows × 8 columns; columns 3-8 are used).
- **Raw Signal Visualization:** Plots all 6 features (gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z) for each sensor with proper labels and color-coding.
- **Scalogram Grid Generation:** Automatically creates the 2x5 grid scalogram image from your upload.
- **Model Inference:** Loads the trained CNN (`best_scalogram_cnn.pth`), predicts the ASL class, and displays a probability bar chart.
- **All steps are performed in-browser.**

### How to Use
1. Place `streamlit_asl_infer.py` and `best_scalogram_cnn.pth` in your project directory.
2. Run the app:
   ```sh
   streamlit run streamlit_asl_infer.py
   ```
3. Open the provided local URL in your browser.
4. Upload a raw `.xlsx` ASL signal file. The app will:
    - Show raw signal plots for all sensors/features.
    - Display the generated scalogram grid image.
    - Predict and display the most likely ASL class with probabilities.

### Inference Requirements
- Python 3.8+
- `streamlit`, `torch`, `numpy`, `pandas`, `matplotlib`, `pywt`, `Pillow`
- Trained model weights: `best_scalogram_cnn.pth`

### Example Output
- Raw signal plots for each sensor, with labeled features.
- 2x5 grid scalogram image.
- Predicted class (A-Z or Rest) and probability bar chart.

See `streamlit_asl_infer.py` for details and customization options.

## Citation
If you use this project for research or deployment, please cite the original dataset source and this repository.

---

## Suggestions and Next Steps
- **Improve 'Rest' class:** Try more advanced augmentation or a dedicated binary classifier for 'Rest' vs. gestures.
- **Regularization:** Add dropout, weight decay, or data augmentation at the image level to further reduce overfitting.
- **Model Variants:** Experiment with deeper or more efficient CNNs (e.g., MobileNet, EfficientNet-lite).
- **Deployment:** Quantize/prune the model for edge deployment, and re-enable ONNX export for TensorRT if needed.
- **Visualization:** Add example scalogram grid images and confusion matrices to this README for clarity.

For questions, improvements, or contributions, please open an issue or pull request.
