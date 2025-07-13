import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import os
import json
from PIL import Image

# Verilerin bulunduğu ana klasör
DATA_DIR = "./images"  # Resimlerin olduğu klasör
LABELS_FILE = "./labels.json"  # Etiketlerin bulunduğu JSON dosyası

# DataLoader için güncellenmiş Dataset sınıfı
class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        with open(labels_file, 'r') as f:
            self.labels_dict = json.load(f)  # JSON dosyasını yükle

        self.image_paths = []
        self.labels = []
        
        for filename, label in self.labels_dict.items():
            img_path = os.path.join(data_dir, filename)
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data augmentation ve normalizasyon işlemleri
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # EfficientNet-B3 için önerilen giriş boyutu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset oluşturma
dataset = CustomDataset(DATA_DIR, LABELS_FILE, transform=transform)

# Eğitim ve doğrulama setlerine ayırma (80% train, 20% val)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# EfficientNet-B3 modelini yükle
model = models.efficientnet_b3(weights='IMAGENET1K_V1')
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),  # Dropout ekleme
    nn.Linear(num_features, 2)
)

# Modeli GPU'ya yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Early Stopping parametreleri
best_val_loss = float('inf')
stopping_patience = 5
patience_counter = 0

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    train_loss = running_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)
    train_prec = precision_score(all_labels, all_preds)
    train_rec = recall_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds)
    
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_prec = precision_score(all_labels, all_preds)
    val_rec = recall_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds)
    
    scheduler.step(val_loss)  # Learning Rate Decay
    
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= stopping_patience:
        print("Early stopping activated!")
        break