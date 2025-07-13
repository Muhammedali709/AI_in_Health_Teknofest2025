import numpy as np
import cv2
import os

# Kırmızı tonlarını tespit etmek için fonksiyon
def is_red_color(pixel, min_red=100, max_green_blue=100):
    red, green, blue = pixel[2], pixel[1], pixel[0]  # OpenCV BGR sırasını RGB'ye çevir
    return red >= min_red and green <= max_green_blue and blue <= max_green_blue

# Etiket dosyalarının kaydedileceği klasör
labels_dir = "./Dataset/labels"

# Sadece hasta görselleri işleyeceğiz
hemorrhage_dir = "./veriler"
hemorrhage_files = [f for f in os.listdir(hemorrhage_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Eğer labels klasörü yoksa oluştur
os.makedirs(labels_dir, exist_ok=True)

# Hasta görsellerini işle
for image_file in hemorrhage_files:
    image_path = os.path.join(hemorrhage_dir, image_file)
    label_file_path = os.path.join(labels_dir, image_file.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt"))
    
    # Görseli yükle
    img = cv2.imread(image_path)
    if img is None:
        print(f"Uyarı: {image_file} açılamadı, atlanıyor.")
        continue
    
    # Görüntü boyutlarını al
    height, width, _ = img.shape

    # Kırmızı pikselleri tespit et
    red_pixels = []

    for y in range(height):
        for x in range(width):
            pixel = img[y, x]
            if is_red_color(pixel):
                red_pixels.append((x, y))

    # Eğer kırmızı pikseller varsa, bounding box hesapla
    if red_pixels:
        x_min = min(p[0] for p in red_pixels)
        y_min = min(p[1] for p in red_pixels)
        x_max = max(p[0] for p in red_pixels)
        y_max = max(p[1] for p in red_pixels)

        # Bounding box merkezi ve boyutlarını hesapla
        x_center = (x_min + x_max) / 2 / width
        y_center = (y_min + y_max) / 2 / height
        bbox_width = (x_max - x_min) / width
        bbox_height = (y_max - y_min) / height

        # Etiket dosyasını oluştur
        with open(label_file_path, 'w') as label_file:
            # class_id 1, çünkü bu kırmızı tonlarını temsil ediyor
            label_file.write(f"1 {x_center} {y_center} {bbox_width} {bbox_height}\n")
        print(f"{image_file} için label dosyası oluşturuldu: {label_file_path}")
    else:
        # Kırmızı pikseller yoksa varsayılan etiket dosyası oluştur
        with open(label_file_path, 'w') as label_file:
            label_file.write("0 0 0 0 0\n")
        print(f"{image_file} için kırmızı tonları tespit edilmedi, varsayılan etiket dosyası oluşturuldu.")

print("Tüm etiket dosyaları oluşturuldu.")