from ultralytics import YOLO
import os
import shutil
import pandas as pd

# Modelin yolunu belirtin (best.pt'nin yolu)
model_path = "./runs/detect/train/weights/last.pt"  # Dosya yolu

# Eğitim için gerekli parametreler
data_yaml = './modelv10m.yaml'  # Veri seti yapılandırması
imgsz = 640  # Resim boyutu
batch_size = 16  # Batch boyutu
epochs = 50  # Eğitim döngü sayısı

# CSV dosyasının yolunu belirtin
csv_file = 'metrics.csv'

# Eğer CSV dosyası yoksa, başlıklarla oluşturun
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Epoch", "Loss", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"])
    df.to_csv(csv_file, index=False)

# Modeli yükleyin (YOLOv8)
model = YOLO(model_path)

# Modelin eğitimine devam etmek için
def custom_train():
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Eğitimi başlat
        results = model.train(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch_size,
            epochs=1,  # Her seferinde bir epoch eğitiyoruz
            save_period=1  # Her epoch sonunda model kaydedilsin
        )

        # Metrikleri elde et
        epoch_metrics = results.metrics
        loss = epoch_metrics.get('loss', None)
        precision = epoch_metrics.get('precision', None)
        recall = epoch_metrics.get('recall', None)
        map50 = epoch_metrics.get('mAP_0.5', None)
        map50_95 = epoch_metrics.get('mAP_0.5:0.95', None)

        # CSV dosyasına ekle
        new_data = {
            "Epoch": epoch + 1,
            "Loss": loss,
            "Precision": precision,
            "Recall": recall,
            "mAP@0.5": map50,
            "mAP@0.5:0.95": map50_95
        }

        # DataFrame'e ekleyip CSV'ye kaydet
        df = pd.DataFrame([new_data])
        df.to_csv(csv_file, mode='a', header=False, index=False)

        # En iyi modeli kaydet
        best_model_path = f"best_epoch_{epoch + 1}.pt"
        shutil.copy('runs/train/exp/weights/best.pt', best_model_path)
        print(f"Best model for epoch {epoch + 1} saved as {best_model_path}")
    
    print(f"Training completed. Metrics saved to {csv_file}.")

# Eğitim fonksiyonunu çağır
custom_train()
