import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from utils import load_data, save_model
from torch.utils.data import DataLoader

# ResNet tabanlı encoder
class ContrastiveEncoder(nn.Module):
    def __init__(self):
        super(ContrastiveEncoder, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()  # Fully connected katmanı çıkar

    def forward(self, x):
        return self.encoder(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContrastiveEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# Sağlıklı ve anormal verileri yükle
data_healthy = load_data("dataset/healthy", transform=transforms.ToTensor())
data_anomaly = load_data("dataset/anomaly", transform=transforms.ToTensor())
dataloader_healthy = DataLoader(data_healthy, batch_size=16, shuffle=True)
dataloader_anomaly = DataLoader(data_anomaly, batch_size=16, shuffle=True)

# Eğitim
num_epochs = 20
for epoch in range(num_epochs):
    for (img1, img2) in zip(dataloader_healthy, dataloader_anomaly):
        img1, img2 = img1.to(device), img2.to(device)
        optimizer.zero_grad()

        feat1 = model(img1)
        feat2 = model(img2)

        loss = criterion(feat1, feat2)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Modeli kaydet
save_model(model, "contrastive_encoder.pth")
print("Contrastive Learning eğitimi tamamlandı.")
