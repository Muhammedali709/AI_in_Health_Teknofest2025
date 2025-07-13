import os
import shutil
from sklearn.model_selection import train_test_split

# Kaynak verinin bulunduğu dizin
source_dir = '../Yolo v8X/Dataset/'

# Hedef dizin (train ve val için)
destination_dir = './Dataset'

# Sağlıklı ve hasta görüntülerin bulunduğu klasörler
healthy_dir = os.path.join(source_dir, 'N_Last')
stroke_dir = os.path.join(source_dir, 'H_Last')

# Eğitim ve doğrulama dizinlerini oluştur
train_healthy_dir = os.path.join(destination_dir, 'train', 'healthy')
val_healthy_dir = os.path.join(destination_dir, 'val', 'healthy')
train_stroke_dir = os.path.join(destination_dir, 'train', 'stroke')
val_stroke_dir = os.path.join(destination_dir, 'val', 'stroke')

# Hedef dizinlerdeki klasörleri oluştur
os.makedirs(train_healthy_dir, exist_ok=True)
os.makedirs(val_healthy_dir, exist_ok=True)
os.makedirs(train_stroke_dir, exist_ok=True)
os.makedirs(val_stroke_dir, exist_ok=True)

# Sağlıklı ve hasta görüntülerin dosya listelerini al
healthy_images = os.listdir(healthy_dir)
stroke_images = os.listdir(stroke_dir)

# Görüntüleri eğitim ve doğrulama setlerine ayır
train_healthy, val_healthy = train_test_split(healthy_images, test_size=0.2, random_state=42)
train_stroke, val_stroke = train_test_split(stroke_images, test_size=0.2, random_state=42)

# Sağlıklı görüntüleri eğitim ve doğrulama setlerine kopyala
for image in train_healthy:
    shutil.copy(os.path.join(healthy_dir, image), train_healthy_dir)

for image in val_healthy:
    shutil.copy(os.path.join(healthy_dir, image), val_healthy_dir)

# Hasta görüntüleri eğitim ve doğrulama setlerine kopyala
for image in train_stroke:
    shutil.copy(os.path.join(stroke_dir, image), train_stroke_dir)

for image in val_stroke:
    shutil.copy(os.path.join(stroke_dir, image), val_stroke_dir)

print("Veri başarıyla ayrıldı ve kopyalandı!")
