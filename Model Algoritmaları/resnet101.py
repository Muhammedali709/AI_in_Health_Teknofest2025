import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import csv
from sklearn.metrics import accuracy_score

# Aygıt seçimi (GPU varsa GPU'yu kullan, yoksa CPU'yu kullan)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Verisetini yükleme ve ön işleme (veri augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 için uygun boyut
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet50 için ortalama ve standart sapma
])

# Verilerin bulunduğu dizin
data_dir = "path_to_your_dataset"  # Burayı veri dizininizle değiştirin

# Veriyi yükle
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Eğitim ve doğrulama veri setlerine ayırma (validation %20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Veri yükleyicilerini oluşturma
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ResNet50 modelini yükleme
model = torchvision.models.resnet50(pretrained=True)

# Son katmanları yeniden yapılandırma (ikili sınıflandırma için)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 sınıf var (İnme var / yok)

# Modeli cihazda çalıştırma
model = model.to(device)

# Kayıp fonksiyonu ve optimizer (AdamW kullanarak)
criterion = nn.CrossEntropyLoss()

# AdamW optimizatörü, weight decay (L2 regularization) ekleyerek
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)  # Daha uygun bir lr ve weight_decay

# Metrikleri CSV dosyasına yazan fonksiyon
def save_metrics_to_csv(epoch, train_loss, val_loss, train_accuracy, val_accuracy, file_name='metrics.csv'):
    # CSV dosyasına metrikleri yazma
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
            writer.writerow(header)  # İlk satır başlıkları yaz
        writer.writerow([epoch, train_loss, val_loss, train_accuracy, val_accuracy])

# Eğitim fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    best_accuracy = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # Eğitim verisi üzerinde döngü
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Gradients sıfırlama
            optimizer.zero_grad()

            # İleri geçiş
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Geriye doğru geçiş
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Doğru tahmin sayısını hesaplama
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        # Eğitim metrikleri
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validasyon verisi üzerinde döngü
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # İleri geçiş
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Doğru tahmin sayısını hesaplama
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        # Validasyon metrikleri
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Epoch başına metrikleri yazdırma
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Metrikleri CSV'ye kaydet
        save_metrics_to_csv(epoch+1, train_loss, val_loss, train_accuracy, val_accuracy)

        # En iyi doğrulama doğruluğunu kaydetme
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")  # En iyi modeli kaydet

    return train_losses, val_losses, train_accuracies, val_accuracies

# Modeli eğitme
train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
