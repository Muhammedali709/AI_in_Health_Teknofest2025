import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from utils import load_data, load_model, visualize_segmentation
from torch.utils.data import DataLoader
from transformers import ViTModel

# ====================
#  ViT + U-Net Modeli
# ====================
class ViTUNet(nn.Module):
    def __init__(self):
        super(ViTUNet, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")  # Önceden eğitilmiş ViT
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vit(x).last_hidden_state  # ViT feature'ları
        x = x.permute(0, 2, 1).view(x.shape[0], 768, 14, 14)  # Şekli U-Net'e uygun hale getir
        x = self.decoder(x)
        return x

# ====================
#  MODEL YÜKLEME
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_unet = ViTUNet().to(device)
contrastive_encoder = load_model("contrastive_encoder.pth").to(device)  # Contrastive Learning modeli
generator = load_model("generator.pth").to(device)  # GAN modeli

# ====================
#  VERİ YÜKLEME
# ====================
data_anomaly = load_data("dataset/anomaly", transform=transforms.ToTensor())
dataloader_anomaly = DataLoader(data_anomaly, batch_size=8, shuffle=True)

# ====================
#  OPTİMİZASYON VE KAYIP FONKSİYONU
# ====================
criterion = nn.BCELoss()
optimizer = optim.Adam(vit_unet.parameters(), lr=0.0002)

# ====================
#  MODEL EĞİTİMİ
# ====================
num_epochs = 20
for epoch in range(num_epochs):
    for images in dataloader_anomaly:
        images = images.to(device)
        optimizer.zero_grad()

        # GAN Rekonstrüksiyon hatasını hesapla
        fake_images = generator(images)
        reconstruction_error = torch.abs(images - fake_images)  # Anomaliyi belirlemek için farkı kullan

        # Contrastive Encoder'dan feature al
        contrastive_features = contrastive_encoder(images)

        # ViT ile segmentasyon
        segment_output = vit_unet(images)

        # Anomalili bölgeleri belirlemek için loss hesapla
        loss = criterion(segment_output, reconstruction_error)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# ====================
#  MODELİ KAYDET
# ====================
torch.save(vit_unet.state_dict(), "vit_unet.pth")
print("ViT Segmentasyon Modeli eğitildi ve kaydedildi.")

# ====================
#  TEST VE GÖRSELLEŞTİRME
# ====================
visualize_segmentation(vit_unet, dataloader_anomaly, device)
