import os
import json

def generate_labels(image_dir, label_dir, output_file):
    labels = {}  # Görsel ismi ve etiketlerin saklanacağı sözlük
    
    # Görsellerin bulunduğu dizindeki tüm dosyaları al
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Her bir görsel için aynı isme sahip etiket dosyasını kontrol et
    for image_file in image_files:
        # Görselin etiket dosyasının yolu
        label_file = os.path.join(label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        if os.path.exists(label_file):
            # Etiket dosyasını oku
            with open(label_file, 'r') as file:
                first_line = file.readline().strip()  # İlk satırı al (ilk karakteri kontrol etmek için)
                
                # Eğer etiket dosyasındaki ilk karakter 1 ise, hasta (1) etiketi ver
                if first_line.startswith('1'):
                    labels[image_file] = 1  # Hasta
                # Eğer etiket dosyasındaki ilk karakter 0 ise, sağlıklı (0) etiketi ver
                elif first_line.startswith('0'):
                    labels[image_file] = 0  # Sağlıklı
                else:
                    print(f"{image_file}: Etiket geçersiz (İlk karakter 0 veya 1 değil)")
        else:
            print(f"{image_file}: Etiket dosyası bulunamadı")
    
    # Sonuçları JSON formatında dosyaya kaydet
    with open(output_file, 'w') as json_file:
        json.dump(labels, json_file, indent=4)
    print(f"Etiketler {output_file} dosyasına kaydedildi.")

# Görsel ve etiket dosyalarının bulunduğu dizinlerin yollarını belirt
image_directory = './images'  # Görsellerin bulunduğu dizin
label_directory = './labels'  # Etiket dosyalarının bulunduğu dizin
output_json_file = 'labels.json'    # Çıktı JSON dosyasının ismi

# Fonksiyonu çalıştır
generate_labels(image_directory, label_directory, output_json_file)
