import os
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO  # YOLOv8 modelini yükle
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

def main():
    # Cihaz seçimi (GPU varsa GPU'yu, yoksa CPU'yu kullan)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Eğitim ve doğrulama parametreleri
    batch_size = 16  # Parametre ismi değişti 'batch' olarak
    learning_rate = 1e-4 # Öğrenim gecikmesi modele bağlı güncellenmelidir.
    epochs = 50

    # Klasörler ve dosya yolları bu dosya yollarını kendi klasörünüze göre güncelleyin.
    model_checkpoints_dir = './model_checkpoints' 
    model_metrics_csv = './model_metrics.csv'

    # Eğer model checkpoint klasörü yoksa oluştur
    if not os.path.exists(model_checkpoints_dir):
        os.makedirs(model_checkpoints_dir)

    # YOLOv8 Small modelini yükleyin
    model = YOLO("yolov8s.pt").to(device)  # 'yolov8s.pt' yolunu geçerli model dosyanızla değiştirin

    # Verisetini yükleyelim
    dataset_yaml = 'modelv8s.yaml'  # YAML dosyasının yolu

    # Metrikler için CSV dosyasını başlat
    metrics_columns = ['epoch', 'accuracy', 'f1_score', 'loss', 'mse', 'rmse']
    metrics_df = pd.DataFrame(columns=metrics_columns)

    # Modeli eğitim için başlat
    model.train(data=dataset_yaml, epochs=epochs, batch=batch_size, device=device, lr0=learning_rate)

    # Eğitim döngüsü
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        true_labels = []
        pred_labels = []

        # Train loader (veri yükleyici) döngüsü
        for images, labels in train_loader:  # DataLoader kullanarak veriyi yükleyin
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # Modelin çıktısını al

            # Loss hesaplama
            loss = model.compute_loss(outputs, labels)  # YOLOv8 için loss hesaplama
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Tahminleri al ve metrikleri hesapla
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

        # Eğitim metrikleri
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        mse = mean_squared_error(true_labels, pred_labels)
        rmse = np.sqrt(mse)

        # Eğitim kaybı ve metrikleri yazdır
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

        # Modeli kaydet
        checkpoint_path = os.path.join(model_checkpoints_dir, f'epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        # Metrikleri CSV'ye kaydet
        metrics_df = metrics_df.append({
            'epoch': epoch + 1,
            'accuracy': accuracy,
            'f1_score': f1,
            'loss': epoch_loss / len(train_loader),
            'mse': mse,
            'rmse': rmse
        }, ignore_index=True)

        # Her epoch sonrası doğrulama metriklerini hesapla
        model.eval()
        val_true_labels = []
        val_pred_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(predicted.cpu().numpy())

        # Doğrulama metrikleri
        val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
        val_f1 = f1_score(val_true_labels, val_pred_labels, average='weighted')
        val_mse = mean_squared_error(val_true_labels, val_pred_labels)
        val_rmse = np.sqrt(val_mse)

        # Doğrulama metriklerini yazdır
        print(f'Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}')

    # Metrikleri CSV dosyasına kaydet
    metrics_df.to_csv(model_metrics_csv, index=False)
    print(f'Model metrics saved to {model_metrics_csv}')

if __name__ == '__main__':
    main()
