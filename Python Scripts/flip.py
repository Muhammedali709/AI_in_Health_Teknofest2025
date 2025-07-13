import cv2
import os

# Giriş ve çıkış klasörlerini belirleyin
input_dir = "./Dataset/H_Last"  # Orijinal görsellerin olduğu klasör
output_dir = "./Dataset/H_Last"  # Ayna görüntülerin kaydedileceği klasör

# Çıkış klasörünü oluştur (varsa atla)
os.makedirs(output_dir, exist_ok=True)

# Görselleri işle
for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    
    # Görüntü formatı kontrolü
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Atlandı: {file_name} (Geçersiz format)")
        continue
    
    # Görüntüyü oku
    image = cv2.imread(input_path)
    if image is None:
        print(f"Hata: {file_name} okunamadı.")
        continue
    
    # Ayna görüntüsünü oluştur
    mirrored_image = cv2.flip(image, 1)
    
    # Yeni dosya adını oluştur (_flipped ekleyerek)
    output_path = os.path.join(output_dir, file_name.replace(".", "_flipped."))
    
    # Yeni görseli kaydet
    cv2.imwrite(output_path, mirrored_image)
    print(f"Ayna görüntüsü oluşturuldu: {output_path}")

print("Ayna görüntü oluşturma işlemi tamamlandı.")
