import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.resnet import resnet101
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================
# 📌 1. Dataset Sınıfı (JPG & PNG Desteği)
# ==========================
class StrokeCTDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    def __len__(self):
        return len(self.image_files)

    def load_bounding_boxes(self, label_path, img_shape):
        h, w = img_shape[:2]
        boxes = []
        labels = []
        
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())

                # Normalize değerleri piksele çevir
                x1 = int((x_center - box_width / 2) * w)
                y1 = int((y_center - box_height / 2) * h)
                x2 = int((x_center + box_width / 2) * w)
                y2 = int((y_center + box_height / 2) * h)

                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))  # Label bilgisi ekle
        
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt"))

        # Görüntüyü yükle
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB formatına çevir
        image = cv2.resize(image, (512, 512))  # Model için boyut ayarı

        # Bounding box'ları yükle
        boxes, labels = self.load_bounding_boxes(label_path, image.shape)

        # PyTorch için tensöre çevir
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image_tensor, target

# ==========================
# 📌 2. Standart Faster R-CNN Modeli (ResNet-101)
# ==========================
def get_standard_faster_rcnn(num_classes):
    # Backbone olarak resnet101 kullan
    backbone = resnet101(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])  # Son katmanı çıkar

    # Faster R-CNN modelini oluştur
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,  
        min_size=512,
        max_size=512
    )

    # Predictör kısmını değiştirelim (box_predictor)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# ==========================
# 📌 3. Modeli Eğitme Kodu
# ==========================
image_dir = "dataset/images"
label_dir = "dataset/labels"

num_epochs = 15  
batch_size = 4
learning_rate = 0.0005  
num_classes = 2  

dataset = StrokeCTDataset(image_dir, label_dir)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_standard_faster_rcnn(num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float("inf")
csv_file = "training_metrics.csv"

# CSV için başlık ekleyelim
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Epoch", "Loss", "Precision", "Recall", "F1_Score"])
    df.to_csv(csv_file, index=False)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        for target in targets:
            all_labels.extend(target["labels"].cpu().numpy())

        with torch.no_grad():
            predictions = model(images)
            for pred in predictions:
                pred_labels = pred["labels"].cpu().numpy()
                all_preds.extend(pred_labels)

    # Metrikleri hesapla
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=1)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=1)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=1)

    avg_loss = epoch_loss / len(data_loader)

    # En iyi modeli kaydet
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best.pt")
        print(f"✅ Yeni en iyi model kaydedildi: best.pt (Loss: {best_loss:.4f})")

    # CSV'ye yaz
    df = pd.DataFrame([[epoch + 1, avg_loss, precision, recall, f1]], columns=["Epoch", "Loss", "Precision", "Recall", "F1_Score"])
    df.to_csv(csv_file, mode="a", header=False, index=False)

    print(f"📊 Epoch {epoch+1}: Loss={avg_loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

print("✅ Eğitim tamamlandı!")
