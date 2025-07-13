import tensorflow as tf
import numpy as np
import os
import csv
import smtplib
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error
from math import sqrt
from tensorflow.keras import layers, Model

# Hyperparameters
batch_size = 32
epochs = 50
latent_dim = 100
image_size = (128, 128, 1)  # Assuming 128x128 grayscale images
learning_rate = 0.0002
data_path = "path_to_your_image_folder"  # Update with your image folder path
best_model_path = "best_gan_model.h5"
csv_file = 'metrics.csv'

# Data loading and preprocessing
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size[:2], color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img)
    return np.array(images)

X_train = load_images_from_folder(data_path)
X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]

# Generator Model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Reshape((1, 1, 256)),
        layers.UpSampling2D(),
        layers.Conv2D(128, kernel_size=3, strides=1, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.UpSampling2D(),
        layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Conv2D(1, kernel_size=3, strides=1, padding='same', activation='tanh')
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=image_size),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN Model (Combining Generator and Discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Build and compile models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate))

# Metrics Calculation
def calculate_metrics(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return recall, precision, accuracy, mse, rmse

# Save metrics to CSV
def save_metrics_to_csv(metrics, epoch):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if epoch == 0:
            writer.writerow(["Epoch", "Discriminator Loss", "Generator Loss", "Discriminator Accuracy", "Generator Accuracy", "Recall", "Precision", "Accuracy", "MSE", "RMSE"])
        writer.writerow([epoch] + metrics)

# Send metrics via email
def send_email(metrics):
    from_email = 'your_email@example.com'
    to_email = 'recipient_email@example.com'
    subject = 'GAN Training Metrics'
    body = f"Metrics: {metrics}"

    # Email server configuration
    server = smtplib.SMTP('smtp.example.com', 587)
    server.starttls()
    server.login(from_email, 'your_email_password')

    message = f"Subject: {subject}\n\n{body}"
    server.sendmail(from_email, to_email, message)
    server.quit()

# Training the GAN
def train_gan(epochs, batch_size):
    half_batch = batch_size // 2
    best_d_loss = float('inf')
    best_g_loss = float('inf')

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_images = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        generated_images = generator.predict(noise)

        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_labels)

        # Calculate metrics
        y_true = np.ones((batch_size, 1))  # True labels for generator output
        y_pred = discriminator.predict(generator.predict(noise))
        recall, precision, accuracy, mse, rmse = calculate_metrics(y_true, y_pred)

        # Log and save the metrics
        metrics = [d_loss[0], g_loss, d_loss[1], g_loss, recall, precision, accuracy, mse, rmse]
        save_metrics_to_csv(metrics, epoch)

        # Update best model
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            generator.save(best_model_path)
            print("Saved best generator model.")

        # Send email after every epoch (optional)
        send_email(metrics)

        # Log the progress
        if epoch % 10 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}] [Recall: {recall}] [Precision: {precision}] [Accuracy: {accuracy}] [MSE: {mse}] [RMSE: {rmse}]")

# Run the training
train_gan(epochs, batch_size)
