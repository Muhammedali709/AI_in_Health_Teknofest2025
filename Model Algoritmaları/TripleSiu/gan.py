import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd
import smtplib
import ssl
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error

def load_data(data_dir, transform):
    dataset = ImageFolder(root=data_dir, transform=transform)
    return dataset

# ====================
#  GAN MODELİ TANIMI
# ====================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# ====================
#  VERİ YÜKLEME
# ====================
data_dir = "./N"
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = load_data(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ====================
#  GAN EĞİTİMİ
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.00015 )
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

# CSV dosyasına kaydetmek için başlıklar
columns = ['Epoch', 'Loss_D', 'Loss_G', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'MSE', 'RMSE']

# CSV dosyasını oluştur
csv_file = "gan_training_metrics.csv"
df = pd.DataFrame(columns=columns)

# E-posta ayarları
sender_email = "sbnpym@gmail.com"
receiver_email = "duncanwalpole21@gmail.com"
app_password = "urzu lpvj lqqf mzpb"

def send_email(subject, body):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("✅ Mail başarıyla gönderildi!")
    except Exception as e:
        print(f"❌ Mail gönderilirken hata oluştu: {e}")

num_epochs = 1000
best_accuracy = 0.0  # En iyi doğruluk başlangıç değeri
best_epoch = 0  # En iyi epoch

for epoch in range(num_epochs):
    epoch_loss_D = 0
    epoch_loss_G = 0
    all_labels = []
    all_preds = []

    for real_images, _ in dataloader:  # _ etiketsiz kısmı alır
        images = real_images.to(device)  # Görselleri cihazınıza gönderin
        real_labels = torch.ones(images.size(0), 1).to(device)  # Gerçek etiketler
        fake_labels = torch.zeros(images.size(0), 1).to(device)  # Sahte etiketler

        optimizer_D.zero_grad()
        outputs = discriminator(images)
        loss_real = criterion(outputs, real_labels)

        fake_images = generator(images)
        outputs = discriminator(fake_images.detach())  # Detach: Graident hesaplamasını engelle
        loss_fake = criterion(outputs, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizer_G.step()

        epoch_loss_D += loss_D.item()
        epoch_loss_G += loss_G.item()

        # Metrik hesaplamaları
        real_labels_np = real_labels.cpu().numpy()
        fake_labels_np = fake_labels.cpu().numpy()
        outputs_np = outputs.cpu().detach().numpy()

        all_labels.extend(real_labels_np)
        all_preds.extend(outputs_np)

    # Metriklerin hesaplanması
    accuracy = accuracy_score(all_labels, np.round(all_preds))
    precision = precision_score(all_labels, np.round(all_preds))
    recall = recall_score(all_labels, np.round(all_preds))
    f1 = f1_score(all_labels, np.round(all_preds))

    mse = mean_squared_error(all_labels, np.round(all_preds))
    rmse = np.sqrt(mse)

    # Epoch sonrasında metrikleri CSV dosyasına yaz
    metrics = {
        'Epoch': epoch + 1,
        'Loss_D': epoch_loss_D / len(dataloader),
        'Loss_G': epoch_loss_G / len(dataloader),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'MSE': mse,
        'RMSE': rmse
    }
    df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

    # CSV dosyasına kaydet
    df.to_csv(csv_file, index=False)

    # Metriklerin mail olarak gönderilmesi
    subject = f"Epoch {epoch+1} - GAN Training Metrics"
    body = f"""
    Epoch: {epoch+1}
    Loss_D: {epoch_loss_D / len(dataloader)}
    Loss_G: {epoch_loss_G / len(dataloader)}
    Accuracy: {accuracy}
    Precision: {precision}
    Recall: {recall}
    F1_Score: {f1}
    MSE: {mse}
    RMSE: {rmse}
    """
    send_email(subject, body)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss_D: {epoch_loss_D / len(dataloader)}, Loss_G: {epoch_loss_G / len(dataloader)}, Accuracy: {accuracy}, F1_Score: {f1}")

    # En iyi modeli kaydet
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch + 1
        # En iyi modelin kaydedilmesi
        torch.save(generator.state_dict(), "best_generator.pth")
        print(f"✅ En iyi model epoch {best_epoch} kaydedildi!")

# Sonuç olarak, en iyi model kaydedildi
print("GAN eğitimi tamamlandı ve model kaydedildi.")
