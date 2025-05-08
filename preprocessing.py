import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

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

# Identify numeric columns (all except one-hot encoded)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# === Step 5: Save the Processed Dataset ===
print(f"[INFO] Saving processed dataset to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)
print("[SUCCESS] Preprocessing complete. File saved.")
