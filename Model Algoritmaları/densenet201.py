import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import csv

# Aygıt seçimi (GPU varsa kullan, yoksa CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Veri ön işleme (ResNet veya DenseNet için uygun)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

# Verilerin bulunduğu dizin
data_dir = "path_to_your_dataset"  # Burayı kendi veri yolunuzla değiştirin

# Veriyi yükleme
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Eğitim ve doğrulama veri setlerini ayırma (%80 eğitim, %20 validasyon)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Veri yükleyicileri oluşturma
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ✅ DenseNet201 modelini yükleme
model = torchvision.models.densenet201(pretrained=True)

# ✅ Son katmanı güncelleme (2 sınıflı çıkış)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)

# Modeli GPU'ya taşı
model = model.to(device)

# ✅ Kayıp fonksiyonu ve AdamW optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00005)

# CSV'ye metrikleri kaydeden fonksiyon
def save_metrics_to_csv(epoch, train_loss, val_loss, train_accuracy, val_accuracy, file_name='metrics.csv'):
    header = ['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy']
    file_exists = False
    try:
        with open(file_name, mode='r') as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    with open(file_name, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)  # İlk satır başlıkları ekle
        writer.writerow([epoch, train_loss, val_loss, train_accuracy, val_accuracy])

# Eğitim fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    best_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds

        # ✅ Validasyon aşaması
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds

        # Epoch başına metrikleri ekrana yazdır
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # ✅ Metrikleri CSV'ye kaydet
        save_metrics_to_csv(epoch+1, train_loss, val_loss, train_accuracy, val_accuracy)

        # ✅ En iyi modeli kaydet
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_densenet201_model.pth")

# ✅ Windows için hata önleme
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
