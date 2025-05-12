import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------
# Step 1: Configurations and Hyperparameters
# --------------------------------------
latent_dim = 100         # Size of random noise input to the generator
batch_size = 128
epochs = 100
learning_rate = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------
# Step 2: Load Preprocessed Data
# --------------------------------------
print("[INFO] Loading preprocessed data...")
df = pd.read_csv("processed_data.csv")
data = df.values.astype(np.float32)

data_tensor = torch.tensor(data)
train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

data_dim = data.shape[1]  # Number of features in each sample

# --------------------------------------
# Step 3: Define Generator and Discriminator
# --------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
            nn.Sigmoid()  # Output between 0 and 1 to match scaled data
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability [0,1]
        )

    def forward(self, x):
        return self.model(x)

# Instantiate models
generator = Generator(latent_dim, data_dim).to(device)
discriminator = Discriminator(data_dim).to(device)

# --------------------------------------
# Step 4: Loss and Optimizers
# --------------------------------------
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# --------------------------------------
# Step 5: Training Loop
# --------------------------------------
print("[INFO] Starting GAN training...")
for epoch in range(epochs):
    g_loss_total = 0
    d_loss_total = 0

    for real_samples, in train_loader:
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # === Train Discriminator ===
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = generator(z)

        real_pred = discriminator(real_samples)
        fake_pred = discriminator(fake_samples.detach())

        d_loss_real = criterion(real_pred, real_labels)
        d_loss_fake = criterion(fake_pred, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # === Train Generator ===
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = generator(z)
        fake_pred = discriminator(fake_samples)

        g_loss = criterion(fake_pred, real_labels)  # Try to fool the discriminator

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()

    print(f"[Epoch {epoch+1}/{epochs}]  D Loss: {d_loss_total:.4f}  G Loss: {g_loss_total:.4f}")

# --------------------------------------
# Step 6: Save Trained Models
# --------------------------------------
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
print("[INFO] Models saved: generator.pth and discriminator.pth")
