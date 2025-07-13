import os
import random
import shutil

# Klasörler
output_images_dir = "./Dataset/output_images2"
labels_dir = "./Dataset/labels"
train_dir = "./Dataset/train"
val_dir = "./Dataset/val"

# Train ve val dizinlerini oluştur
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

# Tüm görselleri al
image_files = [f for f in os.listdir(output_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)  # Rastgele karıştır

# Train/val oranı
val_ratio = 0.2
val_count = int(len(image_files) * val_ratio)
val_files = image_files[:val_count]
train_files = image_files[val_count:]

def move_files(file_list, dest_dir):
    for image_file in file_list:
        image_path = os.path.join(output_images_dir, image_file)
        label_file = image_file.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt")
        label_path = os.path.join(labels_dir, label_file)
        
        # Görseli taşı
        shutil.move(image_path, os.path.join(dest_dir, 'images', image_file))
        
        # Eğer ilgili etiket varsa taşı
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(dest_dir, 'labels', label_file))
        print(f"{image_file} ve {label_file} -> {dest_dir}")

# Train ve val setlerini ayır
move_files(train_files, train_dir)
move_files(val_files, val_dir)

print("Veri seti başarıyla train ve validation olarak ayrıldı.")
