import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification

# Veri Augmentation ve Dönüşümleri
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Rastgele kırpma
    transforms.RandomHorizontalFlip(),  # Yatay çevirme
    transforms.RandomRotation(10),  # Döndürme
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Veri Setini Yükleme
train_dataset = ImageFolder(root="C:/Users/eseda/Desktop/vit/VitData/train", transform=transform)
val_dataset = ImageFolder(root="C:/Users/eseda/Desktop/vit/VitData/val", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


#ViT modeli yükleme
model = ViTForImageClassification.from_pretrained( "facebook/deit-small-patch16-224", num_labels=2, ignore_mismatched_sizes=True)

# Modelin son katmanını sıfırdan oluştur
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.config.hidden_size, 2)
)

# Cihazı belirleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tüm ağırlıkları dondur
for param in model.parameters():
    param.requires_grad = False

# Sadece classifier'ı eğitilebilir yap
for param in model.classifier.parameters():
    param.requires_grad = True

# Optimizasyon ve Kayıp Fonksiyonu
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Eğitim Parametreleri
num_epochs = 10
best_val_acc = 0.0
early_stop_count = 0
patience = 5  # Early Stopping için bekleme süresi

# Eğitim Döngüsü
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images= images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_acc = correct / total
    
    # Validation Aşaması
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
    
    # Early Stopping Mekanizması
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_vit_model.pth")
        early_stop_count = 0
    else:
        early_stop_count += 1
    
    if early_stop_count >= patience:
        print("Early Stopping Uygulandı!")
        break
