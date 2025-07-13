import os
import sys
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO  # YOLOv8 modelini yükle
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

def main():
    # Cihaz seçimi (GPU varsa GPU'yu, yoksa CPU'yu kullan)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Eğitim ve doğrulama parametreleri
    batch_size = 32  # Parametre ismi değişti 'batch' olarak
    learning_rate = 0.003  # Başlangıç learning rate değeri
    epochs = 100  # 100 epoch olarak belirledik
    lr_decay_steps = [20, 40, 60, 80]  # Öğrenme oranını azaltma aşamaları (epoch başına)
    lr_decay_factor = 0.5  # Her aşamada öğrenme oranı ne kadar azalacak

    # Klasörler ve dosya yolları
    model_checkpoints_dir = './model_checkpoints'
    model_metrics_csv = './model_metrics.csv'

    # Eğer model checkpoint klasörü yoksa oluştur
    if not os.path.exists(model_checkpoints_dir):
        os.makedirs(model_checkpoints_dir)

    # YOLOv8 Small modelini yükleyin
    model = YOLO("yolov10m.pt").to(device)  # 'yolov8s.pt' yolunu geçerli model dosyanızla değiştirin

    # Verisetini yükleyelim
    dataset_yaml = './modelv10m.yaml'  # YAML dosyasının yolu

    # Metrikler için CSV dosyasını başlat
    metrics_columns = ['epoch', 'accuracy', 'f1_score', 'loss', 'mse', 'rmse']
    metrics_df = pd.DataFrame(columns=metrics_columns)

    # Modeli eğitim için başlat
    model.train(data=dataset_yaml, epochs=epochs, batch=batch_size, device=device, lr0=learning_rate, imgsz=256)

    # Öğrenme oranı dinamik değişimi için parametreler
    best_loss = float('inf')
    best_val_loss = float('inf')
    no_improvement_count = 0
    best_epoch = 0
    early_stopping_patience = 5  # Early stopping sabrı

    # Modeli eğitim için başlat
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

        # Early stopping kontrolü
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improvement_count = 0
            best_epoch = epoch + 1
            # En iyi modeli kaydet
            torch.save(model.state_dict(), os.path.join(model_checkpoints_dir, 'best.pt'))
        else:
            no_improvement_count += 1

        # Early stopping şartı (5 epoch boyunca kayıp değeri artarsa durdur)
        if no_improvement_count >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

        # Öğrenme oranını kademeli olarak düşür
        # if epoch in lr_decay_steps:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= lr_decay_factor
        #     print(f'Learning rate reduced to {optimizer.param_groups[0]["lr"]:.6f}')

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

        # En iyi doğrulama kaybı durumunda model kaydetme
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), os.path.join(model_checkpoints_dir, 'best_val_model.pt'))
            print(f'Best validation model saved at epoch {epoch+1}')

    # Metrikleri CSV dosyasına kaydet
    metrics_df.to_csv(model_metrics_csv, index=False)
    print(f'Model metrics saved to {model_metrics_csv}')

    # Eğitim tamamlandığında programı kapat
    print(f"Training completed. Best model saved at epoch {best_epoch}.")
    sys.exit()

if __name__ == '__main__':
    main()
