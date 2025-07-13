import os

# Etiket dosyalarının kaydedileceği klasör
labels_dir = "./Dataset/labels"

# Normal (sağlıklı) görselleri işleyeceğiz
normal_dir = "./Dataset/N"
normal_files = [f for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Eğer labels klasörü yoksa oluştur
os.makedirs(labels_dir, exist_ok=True)

# Normal görsellerini işle
for image_file in normal_files:
    image_path = os.path.join(normal_dir, image_file)
    label_file_path = os.path.join(labels_dir, image_file.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt"))
    
    # Her normal görsel için 0 0 0 0 0 içeriğiyle etiket dosyasını oluştur
    with open(label_file_path, 'w') as label_file:
        label_file.write("0 0 0 0 0\n")
    
    print(f"{image_file} için label dosyası oluşturuldu: {label_file_path}")

print("Tüm etiket dosyaları oluşturuldu.")
