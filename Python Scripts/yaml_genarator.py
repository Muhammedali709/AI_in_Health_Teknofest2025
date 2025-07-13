import os
# TÜm dosya yollarını ve sınıf ağırlıklarını kod ve verinize göre güncellemeyi unutmayın!
# Model için YAML dosyasının yolu
yaml_file_path = "./modelv8s.yaml"

# Eğitim ve doğrulama klasörlerinin yolları
train_images_dir = "./train/images"
train_labels_dir = "./train/labels"
val_images_dir = "./val/images"
val_labels_dir = "./val/labels"

# Sınıf sayısı ve sınıf isimleri
nc = 2  # 2 sınıf var: NORMAL ve hemorrhage
class_names = ["NORMAL", "hemorrhage"]

# YAML dosyasını oluşturma
yaml_content = f"""
# YOLOv8 Small model configuration

# Yol verisi
train: {train_images_dir}
val: {val_images_dir}

# Sinif sayisi
nc: {nc}

# Sinif isimleri
names: {class_names}
"""

# YAML dosyasını kaydet
with open(yaml_file_path, "w") as yaml_file:
    yaml_file.write(yaml_content)

print(f"modelv8s.yaml dosyası başarıyla oluşturuldu: {yaml_file_path}")
