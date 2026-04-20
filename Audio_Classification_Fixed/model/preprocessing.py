import os
import librosa
import numpy as np

# ─────────────────────────────────────────────
# CONFIG  – change dataset_path to where your
#           "Voice of Birds" folder lives locally
# ─────────────────────────────────────────────
dataset_path = "data/Voice of Birds"   # <-- update this path

sr            = 22050
n_mels        = 128
fixed_duration = 3.0          # unified with Data_augmentation.py
n_fft         = 2048
hop_length    = 512

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_spectrogram(audio_path):
    """Load an audio file and return a log-mel spectrogram (n_mels × T)."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=fixed_duration)

        # Pad short clips to fixed length
        target_len = int(fixed_duration * sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")

        mel_spec    = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

# ─────────────────────────────────────────────
# Walk dataset and extract features
# ─────────────────────────────────────────────
features_mel = []
labels_mel   = []

for root, _, files in os.walk(dataset_path):
    class_label = os.path.basename(root)
    if class_label == os.path.basename(dataset_path):
        # Skip the root folder itself (no label)
        continue
    for file in files:
        if file.lower().endswith(".mp3"):
            audio_path  = os.path.join(root, file)
            spectrogram = load_spectrogram(audio_path)
            if spectrogram is not None:
                features_mel.append(spectrogram)
                labels_mel.append(class_label)

features_mel = np.array(features_mel)
labels_mel   = np.array(labels_mel)

# Save to model directory (same folder as this script)
out_dir = os.path.dirname(os.path.abspath(__file__))
np.save(os.path.join(out_dir, "features_mel.npy"), features_mel)
np.save(os.path.join(out_dir, "labels_mel.npy"),   labels_mel)

print("Preprocessing Complete!")
print("Feature shape:", features_mel.shape)
print("Labels shape:",  labels_mel.shape)
