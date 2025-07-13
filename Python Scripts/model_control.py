import pandas as pd
import glob
import os

# CSV dosyalarının bulunduğu klasör
csv_folder = "./Model Control/v10/"  # Burayı uygun dizinle değiştirin
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))  # Belirtilen klasördeki tüm CSV dosyalarını al

best_epochs = []  # Her modelin en iyi epoch'una ait metrikleri saklayacağız

for file in csv_files:
    df = pd.read_csv(file)
    
    # En iyi epoch'u mAP50-95'e göre seç
    best_epoch_idx = df['metrics/mAP50-95(B)'].idxmax()
    best_metrics = df.loc[best_epoch_idx, ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']]
    best_metrics['model'] = os.path.basename(file)  # Model adını ekle (sadece dosya adı)
    best_epochs.append(best_metrics)

# Tüm en iyi epoch'ları içeren DataFrame
best_models_df = pd.DataFrame(best_epochs)

# En iyi modeli seç (mAP50-95 değerine göre sıralayıp en üsttekini al)
best_models_df = best_models_df.sort_values(by='metrics/mAP50-95(B)', ascending=False)
best_model = best_models_df.iloc[0]  # En yüksek mAP50-95 değerine sahip modeli seç

# Sonuçları yazdır
print("Her modelin en iyi epoch'ları:")
print(best_models_df)
print("\nEn başarılı model:")
print(best_model)
