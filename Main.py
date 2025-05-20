# GAN-Based Network Traffic Generator and Evaluator (Role 2 + Role 3)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# ---------------- Configuration ----------------
latent_dim = 100
batch_size = 128
more_epochs = 50
learning_rate = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
processed_file = "processed_data.csv"
checkpoint_file = "checkpoint.pth"
scaler_file = "scaler.pkl"
generator_file = "generator.pth"
discriminator_file = "discriminator.pth"
raw_generated_file = "generated_samples.csv"
human_readable_file = "generated_human_readable.csv"

# ---------------- Load and Scale Data ----------------
print("[INFO] Loading and scaling data...")
df = pd.read_csv(processed_file)
columns = df.columns.tolist()

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(df.values.astype(np.float32))
joblib.dump(scaler, scaler_file)

data_tensor = torch.tensor(data_scaled)
train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
data_dim = data_scaled.shape[1]

# ---------------- Define Generator and Discriminator ----------------
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super().__init__()
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

class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
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

# ---------------- Initialize / Load Models ----------------
generator = Generator(latent_dim, data_dim).to(device)
discriminator = Discriminator(data_dim).to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

start_epoch = 0
if os.path.exists(checkpoint_file):
    print(f"[INFO] Resuming training from checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    print("[INFO] Starting training from scratch...")

# ---------------- Training ----------------
criterion = nn.BCELoss()
g_losses, d_losses = [], []

for epoch in range(start_epoch, start_epoch + more_epochs):
    g_loss_total = 0
    d_loss_total = 0

    for i, (real_samples,) in enumerate(train_loader):
        real_samples = real_samples.to(device)
        bs = real_samples.size(0)

        real_labels = torch.ones(bs, 1).uniform_(0.9, 1.0).to(device)
        fake_labels = torch.zeros(bs, 1).uniform_(0.0, 0.1).to(device)

        # Train Generator
        z = torch.randn(bs, latent_dim).to(device)
        fake_samples = generator(z)
        g_loss = criterion(discriminator(fake_samples), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        with torch.no_grad():
            fake_samples = generator(torch.randn(bs, latent_dim).to(device)).detach()
        d_loss = criterion(discriminator(real_samples), real_labels) + \
                 criterion(discriminator(fake_samples), fake_labels)

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    g_losses.append(g_loss_total / len(train_loader))
    d_losses.append(d_loss_total / len(train_loader))

    print(f"[Epoch {epoch+1}] D Loss: {d_losses[-1]:.4f} | G Loss: {g_losses[-1]:.4f}")

    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_file)

# Save final models
torch.save(generator.state_dict(), generator_file)
torch.save(discriminator.state_dict(), discriminator_file)
print("[INFO] Final models saved.")

# ---------------- Generate & Inverse-Scale Samples ----------------
print("[INFO] Generating synthetic data...")
generator.eval()
with torch.no_grad():
    z = torch.randn(1000, latent_dim).to(device)
    fake_scaled = generator(z).cpu().numpy()
    fake_original = scaler.inverse_transform(fake_scaled)
    pd.DataFrame(fake_original, columns=columns).to_csv(raw_generated_file, index=False)

# ---------------- Decode to Human-Readable Format ----------------
def get_onehot_groups(cols, bases):
    return {base: sorted([c for c in cols if c.startswith(base + "_")]) for base in bases}

def reconstruct_ip(row, prefix):
    return ".".join([str(int(round(np.clip(row[f"{prefix}_octet{i}"], 0, 255)))) for i in range(1, 5)])

def decode_onehot(row, group):
    idx = np.argmax(row[group].values)
    return group[idx].split("_", 1)[1]

print("[INFO] Decoding samples...")
df_fake = pd.DataFrame(fake_original, columns=columns)
groups = get_onehot_groups(columns, ["proto", "conn_state", "history", "label"])

human_rows = []
for _, row in df_fake.iterrows():
    decoded = {
        "id.orig_h": reconstruct_ip(row, "orig_ip"),
        "id.resp_h": reconstruct_ip(row, "resp_ip"),
        "id.orig_p": int(row["id.orig_p"]),
        "id.resp_p": int(row["id.resp_p"]),
        "proto": decode_onehot(row, groups["proto"]),
        "conn_state": decode_onehot(row, groups["conn_state"]),
        "missed_bytes": int(row["missed_bytes"]),
        "history": decode_onehot(row, groups["history"]),
        "orig_pkts": int(row["orig_pkts"]),
        "orig_ip_bytes": int(row["orig_ip_bytes"]),
        "resp_pkts": int(row["resp_pkts"]),
        "resp_ip_bytes": int(row["resp_ip_bytes"]),
        "label": decode_onehot(row, groups["label"])
    }
    human_rows.append(decoded)

df_human = pd.DataFrame(human_rows)
df_human.to_csv(human_readable_file, index=False)
print(f"[SUCCESS] Human-readable synthetic data saved to: {human_readable_file}")

# ---------------- Plot Loss & Distributions ----------------
plt.figure()
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("GAN Training Loss")
#plt.savefig("training_loss_plot.png")
plt.show()

print("[INFO] Comparing real vs generated distributions...")
df_real = pd.read_csv(processed_file)
df_real_inv = pd.DataFrame(scaler.inverse_transform(df_real), columns=df_real.columns)
df_real_inv["proto"] = df_real[groups["proto"]].idxmax(axis=1).str.replace("proto_", "")
df_real_inv["label"] = df_real[groups["label"]].idxmax(axis=1).str.replace("label_", "")

def plot_feature_dist(real, fake, feature):
    plt.figure(figsize=(6,4))
    plt.hist(real[feature], bins=30, alpha=0.6, label="Real", density=True)
    plt.hist(fake[feature], bins=30, alpha=0.6, label="Generated", density=True)
    plt.title(f"{feature} Distribution (Normalized)")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(f"dist_{feature}.png")
    plt.show()

def plot_category_dist(real, fake, feature):
    r_counts = real[feature].value_counts(normalize=True).sort_index()
    f_counts = fake[feature].value_counts(normalize=True).sort_index()
    all_keys = sorted(set(r_counts.index).union(f_counts.index))

    r_vals = [r_counts.get(k, 0) * 100 for k in all_keys]
    f_vals = [f_counts.get(k, 0) * 100 for k in all_keys]

    x = np.arange(len(all_keys))
    plt.figure(figsize=(6,4))
    plt.bar(x - 0.2, r_vals, width=0.4, label="Real")
    plt.bar(x + 0.2, f_vals, width=0.4, label="Generated")
    plt.xticks(x, all_keys, rotation=45)
    plt.ylabel("Percentage (%)")
    plt.title(f"{feature} Category Distribution (Normalized)")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"dist_{feature}.png")
    plt.show()

plot_feature_dist(df_real_inv, df_human, "orig_ip_bytes")
plot_feature_dist(df_real_inv, df_human, "resp_ip_bytes")
plot_category_dist(df_real_inv, df_human, "proto")
plot_category_dist(df_real_inv, df_human, "label")

#print("[✅] All stages complete: training, generation, decoding, and evaluation.")
