import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
# Load features saved by preprocessing.py
# ─────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
features_mel = np.load(os.path.join(script_dir, "features_mel.npy"), allow_pickle=True)
labels_mel   = np.load(os.path.join(script_dir, "labels_mel.npy"),   allow_pickle=True)

# ─────────────────────────────────────────────
# Encode labels → one-hot
# ─────────────────────────────────────────────
label_encoder   = LabelEncoder()
labels_encoded  = label_encoder.fit_transform(labels_mel)
num_classes     = len(label_encoder.classes_)
labels_onehot   = to_categorical(labels_encoded, num_classes=num_classes)

# Add channel dimension: (N, n_mels, T) → (N, n_mels, T, 1)
features_mel = features_mel[..., np.newaxis]

# ─────────────────────────────────────────────
# Train / Val / Test  split  (70 / 15 / 15)
# ─────────────────────────────────────────────
x_train, x_temp, y_train, y_temp = train_test_split(
    features_mel, labels_onehot, test_size=0.30, random_state=42
)

# stratify the second split on the temp labels argmax
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.50, random_state=42
)

# ─────────────────────────────────────────────
# Save splits
# ─────────────────────────────────────────────
np.save(os.path.join(script_dir, "x_train.npy"), x_train)
np.save(os.path.join(script_dir, "y_train.npy"), y_train)
np.save(os.path.join(script_dir, "x_val.npy"),   x_val)
np.save(os.path.join(script_dir, "y_val.npy"),   y_val)
np.save(os.path.join(script_dir, "x_test.npy"),  x_test)
np.save(os.path.join(script_dir, "y_test.npy"),  y_test)
np.save(os.path.join(script_dir, "label_encoder.npy"), label_encoder.classes_)

print("Data Preparation Complete!")
print(f"Number of classes : {num_classes}")
print(f"Train shape       : {x_train.shape}  {y_train.shape}")
print(f"Validation shape  : {x_val.shape}    {y_val.shape}")
print(f"Test shape        : {x_test.shape}   {y_test.shape}")
