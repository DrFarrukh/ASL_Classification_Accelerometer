import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Config ----
DATA_ROOT = 'CombinedScalogramsGrid'  # or 'Scalograms/gyro'
IMG_SIZE = 128  # resize for Jetson Nano efficiency
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 27  # A-Z + Rest
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---- Dataset ----
class ScalogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(root_dir)))}
        for cls in self.class_to_idx:
            files = glob(os.path.join(root_dir, cls, '*.png'))
            self.samples.extend(files)
            self.labels.extend([self.class_to_idx[cls]] * len(files))
        self.transform = transform
        logging.info(f"Dataset initialized with {len(self.samples)} samples")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def get_loaders(data_root, img_size, batch_size):
    logging.info(f"Preparing data loaders with root: {data_root}, img_size: {img_size}, batch_size: {batch_size}")
    dataset = ScalogramDataset(data_root, transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]))
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.15, stratify=dataset.labels, random_state=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    logging.info(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    return train_loader, val_loader, dataset.class_to_idx

# ---- Lightweight CNN ----
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
        logging.info(f"Initialized SimpleJetsonCNN with {num_classes} output classes")
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.close()

# ---- Training Loop ----
def train():
    logging.info("Starting training process")
    train_loader, val_loader, class_to_idx = get_loaders(DATA_ROOT, IMG_SIZE, BATCH_SIZE)
    model = SimpleJetsonCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    for epoch in range(EPOCHS):
        model.train()
        total, correct, train_loss = 0, 0, 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
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
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validation
        model.eval()
        total, correct, val_loss = 0, 0, 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = correct / total
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_scalogram_cnn.pth')
            logging.info(f"New best model saved with validation accuracy: {best_acc:.4f}")
    
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
    plt.savefig('training_curves.png')
    plt.close()
    
    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, list(class_to_idx.keys()))
    cr = classification_report(all_labels, all_preds, target_names=list(class_to_idx.keys()))
    with open('scalogram_classification_report.txt', 'w') as f:
        f.write(cr)
    logging.info("Classification Report:\n" + cr)
    

if __name__ == '__main__':
    logging.info(f"Using device: {DEVICE}")
    train()
