import os
import cv2
import numpy as np
from random import randint

# CutMix fonksiyonu
def cutmix(image1, image2, beta=1.0):
    h, w, _ = image1.shape
    lam = np.random.beta(beta, beta)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = randint(0, w - cut_w)
    cy = randint(0, h - cut_h)
    image2_cut = image2[cy:cy + cut_h, cx:cx + cut_w]
    image1[cy:cy + cut_h, cx:cx + cut_w] = image2_cut
    return image1

# Görselleri yükleme fonksiyonu
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (256, 256))  # Görselleri 256x256 boyutuna getir
                images.append(img)
    return images

# CutMix uygula ve kaydet
def apply_cutmix_and_save(image_folder, output_folder, num_cutmix_per_image=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Çıktı klasörü oluştur

    # Görselleri yükle
    images = load_images_from_folder(image_folder)
    num_images = len(images)

    # Her bir görsel için rastgele CutMix uygula
    for i in range(num_images):
        for k in range(num_cutmix_per_image):
            # Rastgele bir görsel seç (kendisi hariç)
            j = np.random.choice(num_images)
            while j == i:
                j = np.random.choice(num_images)

            image1 = images[i].copy()
            image2 = images[j].copy()

            # CutMix uygula
            augmented_image = cutmix(image1, image2)

            # Kaydet
            output_path = os.path.join(output_folder, f'cutmix_{i}_{j}_{k}.jpg')
            cv2.imwrite(output_path, augmented_image)
            print(f"CutMix image saved: {output_path}")

# Ana kod
if __name__ == "__main__":
    image_folder = '../output_images'  # Görsellerin bulunduğu klasör
    output_folder = '../cutmix_images'  # CutMix görsellerinin kaydedileceği klasör

    # CutMix uygula ve kaydet
    apply_cutmix_and_save(image_folder, output_folder, num_cutmix_per_image=2)