"""
EDA2.py – Raw-audio EDA
Reads audio files directly from disk (no pre-extracted npy required).
Shows:
  1. Class distribution bar chart.
  2. Mel-spectrogram of one randomly chosen audio file.
"""
import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# CONFIG – update to your local dataset folder
# ─────────────────────────────────────────────
dataset_path = "data/Voice of Birds"   # <-- update this

# ─────────────────────────────────────────────
# Count files per class
# ─────────────────────────────────────────────
class_counts = {}
for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        num_files = len([f for f in os.listdir(folder_path)
                         if f.lower().endswith(".mp3")])
        if num_files > 0:
            class_counts[folder] = num_files

if not class_counts:
    raise FileNotFoundError(
        f"No subfolders with .mp3 files found in '{dataset_path}'. "
        "Please update 'dataset_path'."
    )

# ─────────────────────────────────────────────
# 1. Class distribution
# ─────────────────────────────────────────────
plt.figure(figsize=(max(10, len(class_counts) * 0.6), 5))
plt.bar(class_counts.keys(), class_counts.values(), color="skyblue", edgecolor="navy")
plt.xlabel("Bird Species (Classes)")
plt.ylabel("Number of Audio Files")
plt.title("Distribution of Audio Files per Class")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 2. Random sample mel-spectrogram
# ─────────────────────────────────────────────
random_class  = random.choice(list(class_counts.keys()))
random_folder = os.path.join(dataset_path, random_class)
mp3_files     = [f for f in os.listdir(random_folder) if f.lower().endswith(".mp3")]
random_file   = random.choice(mp3_files)
random_path   = os.path.join(random_folder, random_file)

print(f"Plotting: {random_class} / {random_file}")

y, sr      = librosa.load(random_path, sr=22050)
mel_spec   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048,
                                             hop_length=512, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(8, 4))
librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512,
                         x_axis="time", y_axis="mel", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title(f"Mel Spectrogram – {random_class}")
plt.tight_layout()
plt.show()
