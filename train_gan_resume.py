import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# ---------------- Configuration ----------------
latent_dim = 100
batch_size = 128
more_epochs = 50
learning_rate = 0.0002
checkpoint_file = "checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Data ----------------
df = pd.read_csv("processed_data.csv")
data = df.values.astype(np.float32)

# IMPORTANT: Make sure data is scaled to [-1, 1] for Tanh
data_tensor = torch.tensor(data)
train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
data_dim = data.shape[1]

# ---------------- Define Generator ----------------
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
            nn.Tanh()  # Assuming data is scaled to [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# ---------------- Define Discriminator ----------------
class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ---------------- Initialize Models ----------------
generator = Generator(latent_dim, data_dim).to(device)
discriminator = Discriminator(data_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# ---------------- Resume from Checkpoint if Available ----------------
start_epoch = 0
if os.path.exists(checkpoint_file):
    print(f"[INFO] Resuming training from {checkpoint_file}...")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    except RuntimeError as e:
        print(f"[WARNING] Failed to load checkpoint due to architecture mismatch. Starting from scratch.\n{e}")
        start_epoch = 0
else:
    print("[INFO] Starting training from scratch...")

# ---------------- Training Loop ----------------
criterion = nn.BCELoss()
update_d_every = 2  # Train discriminator every 2 batches

for epoch in range(start_epoch, start_epoch + more_epochs):
    g_loss_total = 0
    d_loss_total = 0

    for i, (real_samples,) in enumerate(train_loader):
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        # Label smoothing and noise
        real_labels = torch.ones(batch_size, 1).uniform_(0.9, 1.0).to(device)
        fake_labels = torch.zeros(batch_size, 1).uniform_(0.0, 0.1).to(device)

        # === Train Generator ===
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = generator(z)
        fake_pred = discriminator(fake_samples)
        g_loss = criterion(fake_pred, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        g_loss_total += g_loss.item()

        # === Train Discriminator (less frequently) ===
        if i % update_d_every == 0:
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = generator(z).detach()

            real_pred = discriminator(real_samples)
            fake_pred = discriminator(fake_samples)

            d_loss = criterion(real_pred, real_labels) + criterion(fake_pred, fake_labels)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            d_loss_total += d_loss.item()

    avg_d_loss = d_loss_total / len(train_loader)
    avg_g_loss = g_loss_total / len(train_loader)

    print(f"[Epoch {epoch+1}] D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_file)

# ---------------- Save Final Models ----------------
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
print("[INFO] Final models saved.")
