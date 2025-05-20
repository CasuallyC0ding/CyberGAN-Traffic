import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === Step 1: Load and Sample the Dataset ===
INPUT_FILE = "Dataset.csv"
OUTPUT_FILE = "processed_data.csv"
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

print("[INFO] Loading dataset...")
df = pd.read_csv(INPUT_FILE)
df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

# === Step 2: Split IP Addresses into Octets ===
def split_ip_column(ip_series, prefix):
    octets = ip_series.str.split('.', expand=True).astype(float)
    octets.columns = [f"{prefix}_octet{i+1}" for i in range(4)]
    return octets

print("[INFO] Splitting IP addresses...")
df_orig_ip = split_ip_column(df["id.orig_h"], "orig_ip")
df_resp_ip = split_ip_column(df["id.resp_h"], "resp_ip")

df = pd.concat([df, df_orig_ip, df_resp_ip], axis=1)
df.drop(columns=["id.orig_h", "id.resp_h"], inplace=True)

# === Step 3: One-Hot Encode Categorical Features ===
print("[INFO] One-hot encoding categorical columns...")
categorical_cols = ["proto", "conn_state", "history", "label"]
df = pd.get_dummies(df, columns=categorical_cols)

# === Step 4: Scale Numerical Features ===
print("[INFO] Scaling numerical features...")
scaler = MinMaxScaler()

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# === Step 5: Save the Processed Dataset ===
print(f"[INFO] Saving processed dataset to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)
print("[SUCCESS] Preprocessing complete. File saved.")

# ---------------- Configuration ----------------
latent_dim = 100
batch_size = 128
more_epochs = 50
learning_rate = 0.0002
checkpoint_file = "checkpoint.pth"
generated_file = "generated_samples.csv"
scaler_file = "scaler.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load and Preprocess Data ----------------
df = pd.read_csv("processed_data.csv")

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(df.values.astype(np.float32))

import joblib
joblib.dump(scaler, scaler_file)

data_tensor = torch.tensor(data_scaled)
train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
data_dim = data_scaled.shape[1]

# ---------------- Generator ----------------
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
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# ---------------- Discriminator ----------------
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

# ---------------- Init Models ----------------
generator = Generator(latent_dim, data_dim).to(device)
discriminator = Discriminator(data_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# ---------------- Resume from Checkpoint ----------------
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
        print(f"[WARNING] Checkpoint failed to load. Starting fresh.\n{e}")
        start_epoch = 0
else:
    print("[INFO] Starting training from scratch...")

# ---------------- Training ----------------
criterion = nn.BCELoss()
update_d_every = 2
g_losses, d_losses = [], []

for epoch in range(start_epoch, start_epoch + more_epochs):
    g_loss_total = 0
    d_loss_total = 0

    for i, (real_samples,) in enumerate(train_loader):
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        real_labels = torch.ones(batch_size, 1).uniform_(0.9, 1.0).to(device)
        fake_labels = torch.zeros(batch_size, 1).uniform_(0.0, 0.1).to(device)

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = generator(z)
        fake_pred = discriminator(fake_samples)
        g_loss = criterion(fake_pred, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        g_loss_total += g_loss.item()

        # Train Discriminator
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
    d_losses.append(avg_d_loss)
    g_losses.append(avg_g_loss)

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

# ---------------- Generate & Save Synthetic Samples (Human-Readable) ----------------
print("[INFO] Generating synthetic samples...")
generator.eval()
with torch.no_grad():
    z = torch.randn(1000, latent_dim).to(device)
    fake_samples = generator(z).cpu().numpy()

scaler = joblib.load(scaler_file)
fake_original_scale = scaler.inverse_transform(fake_samples)
fake_df = pd.DataFrame(fake_original_scale, columns=df.columns)

# Recombine IP addresses
def recombine_ip(df, prefix):
    octets = [df[f"{prefix}_octet{i+1}"].round().astype(int).clip(0, 255).astype(str) for i in range(4)]
    return octets[0] + '.' + octets[1] + '.' + octets[2] + '.' + octets[3]

fake_df["id.orig_h"] = recombine_ip(fake_df, "orig_ip")
fake_df["id.resp_h"] = recombine_ip(fake_df, "resp_ip")
fake_df.drop(columns=[col for col in fake_df.columns if "octet" in col], inplace=True)

# Reverse one-hot encoding
def reverse_one_hot(df, prefix):
    one_hot_cols = [col for col in df.columns if col.startswith(prefix + "_")]
    if not one_hot_cols:
        return df
    idx = df[one_hot_cols].values.argmax(axis=1)
    labels = [col.split(prefix + "_", 1)[-1] for col in one_hot_cols]
    df[prefix] = [labels[i] for i in idx]
    df.drop(columns=one_hot_cols, inplace=True)
    return df

for prefix in ["proto", "conn_state", "history", "label"]:
    fake_df = reverse_one_hot(fake_df, prefix)

fake_df.to_csv(generated_file, index=False)
print(f"[INFO] Human-readable generated samples saved to {generated_file}")

# ---------------- Plot Training Loss ----------------
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_plot.png")
plt.show()

print("[INFO] Training insights and synthetic data generated.")
