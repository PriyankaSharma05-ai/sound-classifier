import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG – must match preprocessing.py
# ─────────────────────────────────────────────
sr             = 22050
n_mels         = 128
fixed_duration = 3.0          # same as preprocessing.py
n_fft          = 2048
hop_length     = 512

# ─────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────
def add_noise(y, noise_factor=0.005):
    """Add Gaussian white noise."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_stretch(y, rate=1.1):
    """Speed up or slow down without changing pitch."""
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, n_steps=2):
    """Shift pitch by n semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def to_spectrogram(y):
    """Convert waveform → log-mel spectrogram."""
    target_len = int(fixed_duration * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    mel  = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                           n_fft=n_fft, hop_length=hop_length)
    return librosa.power_to_db(mel, ref=np.max)

def augment_audio(audio_path):
    """Return a list of (spectrogram, tag) tuples for one audio file."""
    y, _ = librosa.load(audio_path, sr=sr, duration=fixed_duration)
    results = []
    try:
        results.append((to_spectrogram(y),                     "original"))
        results.append((to_spectrogram(add_noise(y)),           "noise"))
        results.append((to_spectrogram(time_stretch(y, 1.1)),   "stretch_fast"))
        results.append((to_spectrogram(time_stretch(y, 0.9)),   "stretch_slow"))
        results.append((to_spectrogram(pitch_shift(y, sr, 2)),  "pitch_up"))
        results.append((to_spectrogram(pitch_shift(y, sr, -2)), "pitch_down"))
    except Exception as e:
        print(f"Augmentation error for {audio_path}: {e}")
    return results

# ─────────────────────────────────────────────
# Demo: visualise augmentations for one file
# ─────────────────────────────────────────────
def demo_augmentation(audio_path):
    """Plot all augmented spectrograms side-by-side for one clip."""
    variants = augment_audio(audio_path)
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (spec, tag) in zip(axes, variants):
        librosa.display.specshow(spec, sr=sr, hop_length=hop_length,
                                 x_axis="time", y_axis="mel", cmap="magma", ax=ax)
        ax.set_title(tag)
        ax.axis("off")
    plt.suptitle("Augmentation variants", fontsize=14)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# Full-dataset augmentation (optional)
# ─────────────────────────────────────────────
def augment_dataset(dataset_path, out_dir=None):
    """
    Walk dataset_path, augment every .mp3, and save augmented
    features_mel_aug.npy / labels_mel_aug.npy alongside the originals.
    """
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    aug_features, aug_labels = [], []

    for root, _, files in os.walk(dataset_path):
        class_label = os.path.basename(root)
        if class_label == os.path.basename(dataset_path):
            continue
        for file in files:
            if file.lower().endswith(".mp3"):
                path = os.path.join(root, file)
                for spec, _ in augment_audio(path):
                    aug_features.append(spec)
                    aug_labels.append(class_label)

    aug_features = np.array(aug_features)
    aug_labels   = np.array(aug_labels)

    np.save(os.path.join(out_dir, "features_mel_aug.npy"), aug_features)
    np.save(os.path.join(out_dir, "labels_mel_aug.npy"),   aug_labels)
    print("Augmentation complete!")
    print("Augmented feature shape:", aug_features.shape)
    return aug_features, aug_labels


if __name__ == "__main__":
    # Quick demo – update path to any .mp3 on your machine
    sample_path = "data/Voice of Birds/some_class/sample.mp3"
    if os.path.exists(sample_path):
        demo_augmentation(sample_path)
    else:
        print("Update 'sample_path' to a real .mp3 to see the demo.")
