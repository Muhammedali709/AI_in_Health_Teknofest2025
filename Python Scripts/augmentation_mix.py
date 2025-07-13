import os
import cv2
import numpy as np
import random

# 📌 Gaussian Blur fonksiyonu
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# 📌 Scale (Ölçeklendirme) fonksiyonu
def apply_scale(image, scale_factor=1.2):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    final_image = cv2.resize(resized_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    return final_image

# 📌 Görselleri yükleme fonksiyonu
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (256, 256))  # Tüm görselleri 256x256 boyutuna getir
                images.append(img)
                filenames.append(filename)
    return images, filenames

# 📌 Veriyi belirtilen oranlarda işleyip kaydetme fonksiyonu
def process_and_save_images(image_folder, output_folder, scale_factor=1.2, original_ratio=0.5, blur_ratio=0.25, scale_ratio=0.25):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Çıktı klasörünü oluştur

    original_folder = os.path.join(output_folder, "original")
    blur_folder = os.path.join(output_folder, "blur")
    scale_folder = os.path.join(output_folder, "scale")

    # Klasörleri oluştur
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(blur_folder, exist_ok=True)
    os.makedirs(scale_folder, exist_ok=True)

    # Görselleri yükle ve karıştır
    images, filenames = load_images_from_folder(image_folder)
    data = list(zip(images, filenames))
    random.shuffle(data)  # Karıştırarak overfitting'i önlüyoruz

    total_images = len(data)
    
    # Oranlara göre kaç tane veri ayrılacağını hesapla
    num_original = int(total_images * original_ratio)
    num_blur = int(total_images * blur_ratio)
    num_scale = total_images - (num_original + num_blur)  # Kalanları scale için kullan

    print(f"Total Images: {total_images}")
    print(f"Original: {num_original}, Blur: {num_blur}, Scale: {num_scale}")

    # 📌 Orijinal verileri kaydet
    for i, (image, filename) in enumerate(data[:num_original]):
        output_path = os.path.join(original_folder, f'original_{i}_{filename}')
        cv2.imwrite(output_path, image)
        print(f"Original image saved: {output_path}")

    # 📌 Gaussian Blur uygulanmış verileri kaydet
    for i, (image, filename) in enumerate(data[num_original:num_original + num_blur]):
        blurred_image = apply_gaussian_blur(image)
        output_path = os.path.join(blur_folder, f'blur_{i}_{filename}')
        cv2.imwrite(output_path, blurred_image)
        print(f"Blurred image saved: {output_path}")

    # 📌 Scale uygulanmış verileri kaydet
    for i, (image, filename) in enumerate(data[num_original + num_blur:]):
        scaled_image = apply_scale(image, scale_factor=scale_factor)
        output_path = os.path.join(scale_folder, f'scale_{i}_{filename}')
        cv2.imwrite(output_path, scaled_image)
        print(f"Scaled image saved: {output_path}")

# 📌 Ana kod
if __name__ == "__main__":
    image_folder = '../Our Alghorithm/Data/hemorrhagic_rotated_images'  # İşlenecek görsellerin olduğu klasör
    output_folder = '../Our Alghorithm/Data/H'  # Tüm çıktılar burada saklanacak

    # Belirtilen oranlarda işlem yap ve kaydet
    process_and_save_images(image_folder, output_folder, scale_factor=1.2, original_ratio=0.5, blur_ratio=0.25, scale_ratio=0.25)
