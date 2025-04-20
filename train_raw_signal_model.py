import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ---- Config ----
DATA_PATH = 'asl_upsampled_augmented.npz'
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 27
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATIENCE = 10  # Number of epochs to wait for improvement before early stopping

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---- Dataset ----
class RawASLDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.arrays = data['arrays']  # shape: (N, 5, 90, 6)
        self.labels = data['labels']
        # Create class_to_idx mapping
        unique_classes = sorted(set(self.labels))
        self.class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        self.targets = np.array([self.class_to_idx[l] for l in self.labels], dtype=np.int64)

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        x = self.arrays[idx].astype(np.float32)  # (5, 90, 6)
        # Rearrange to (channels=6, depth=5, height=1, width=90)
        x = np.transpose(x, (2, 0, 1))  # (6, 5, 90)
        x = np.expand_dims(x, axis=2)  # (6, 5, 1, 90)
        x = torch.from_numpy(x)
        return x, torch.tensor(self.targets[idx], dtype=torch.long)

def get_loaders(npz_path, batch_size):
    dataset = RawASLDataset(npz_path)
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.15, stratify=dataset.targets, random_state=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    logging.info(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    return train_loader, val_loader, dataset.class_to_idx

# ---- Lightweight 3D CNN ----
class RawSignalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=(3,1,5), padding=(1,0,2)),  # (channels, depth, height, width)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,1,3), padding=(1,0,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 6, 5, 1, 90)
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---- Training Loop ----
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('rawsignal_confusion_matrix.png')
    plt.close()

def train():
    logging.info("Starting training process for raw signal model")
    train_loader, val_loader, class_to_idx = get_loaders(DATA_PATH, BATCH_SIZE)
    model = RawSignalCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        total, correct, train_loss = 0, 0, 0.0
        for i, (x, labels) in enumerate(train_loader):
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            train_loss += loss.item()
            if i % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{EPOCHS} - Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")
        train_acc = correct / total
        train_loss /= len(train_loader)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        # Validation
        model.eval()
        total, correct, val_loss = 0, 0, 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                outputs = model(x)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = correct / total
        val_loss /= len(val_loader)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_rawsignal_cnn.pth')
            logging.info(f"New best model saved with validation accuracy: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    logging.info(f'Training complete. Best Validation Accuracy: {best_acc:.4f}')
    # Plot training curves
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.savefig('rawsignal_training_curves.png')
    plt.close()
    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, list(class_to_idx.keys()))
    cr = classification_report(all_labels, all_preds, target_names=list(class_to_idx.keys()))
    with open('rawsignal_classification_report.txt', 'w') as f:
        f.write(cr)
    logging.info("Classification Report:\n" + cr)

if __name__ == '__main__':
    logging.info(f"Using device: {DEVICE}")
    train()
