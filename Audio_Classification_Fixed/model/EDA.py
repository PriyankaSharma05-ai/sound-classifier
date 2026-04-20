"""
EDA.py – Exploratory Data Analysis
Loads pre-extracted mel-spectrograms and shows:
  1. One spectrogram per class.
  2. Bar chart of samples per class.
"""
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Load saved features
# ─────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
features_mel = np.load(os.path.join(script_dir, "features_mel.npy"), allow_pickle=True)
labels_mel   = np.load(os.path.join(script_dir, "labels_mel.npy"),   allow_pickle=True)

sr         = 22050
hop_length = 512

# ─────────────────────────────────────────────
# 1. One spectrogram per class
# ─────────────────────────────────────────────
def plot_one_per_class(features, labels):
    unique_classes = np.unique(labels)
    num_classes    = len(unique_classes)

    fig, axes = plt.subplots(num_classes, 1, figsize=(8, 2.5 * num_classes))
    if num_classes == 1:
        axes = [axes]   # make iterable

    for ax, class_label in zip(axes, unique_classes):
        idx = np.where(labels == class_label)[0][0]
        librosa.display.specshow(features[idx], sr=sr, hop_length=hop_length,
                                 x_axis="time", y_axis="mel",
                                 cmap="magma", ax=ax)
        ax.set_title(f"Class: {class_label}")
        ax.axis("off")

    plt.suptitle("One Mel-Spectrogram per Class", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# 2. Class distribution bar chart
# ─────────────────────────────────────────────
def plot_class_distribution(labels):
    from collections import Counter
    counts = Counter(labels)
    classes = sorted(counts.keys())
    values  = [counts[c] for c in classes]

    plt.figure(figsize=(max(10, len(classes) * 0.6), 5))
    plt.bar(classes, values, color="skyblue", edgecolor="navy")
    plt.xlabel("Bird Species (Classes)")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Audio Samples per Class")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
print(f"Total samples : {len(labels_mel)}")
print(f"Unique classes: {len(np.unique(labels_mel))}")
print(f"Feature shape : {features_mel.shape}")

plot_one_per_class(features_mel, labels_mel)
plot_class_distribution(labels_mel)
