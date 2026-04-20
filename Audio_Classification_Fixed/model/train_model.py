"""
train_model.py – CNN Audio Classifier
Loads the prepared data splits, builds a CNN, trains it, evaluates it,
and saves the trained model.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
# Load splits
# ─────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

x_train = np.load(os.path.join(script_dir, "x_train.npy"))
y_train = np.load(os.path.join(script_dir, "y_train.npy"))
x_val   = np.load(os.path.join(script_dir, "x_val.npy"))
y_val   = np.load(os.path.join(script_dir, "y_val.npy"))
x_test  = np.load(os.path.join(script_dir, "x_test.npy"))
y_test  = np.load(os.path.join(script_dir, "y_test.npy"))
label_classes = np.load(os.path.join(script_dir, "label_encoder.npy"), allow_pickle=True)

input_shape = x_train.shape[1:]   # (n_mels, T, 1)
num_classes = y_train.shape[1]

print(f"Input shape : {input_shape}")
print(f"Num classes : {num_classes}")
print(f"Train size  : {len(x_train)}")

# ─────────────────────────────────────────────
# Normalize to [0, 1]
# ─────────────────────────────────────────────
x_min, x_max = x_train.min(), x_train.max()
x_train = (x_train - x_min) / (x_max - x_min + 1e-8)
x_val   = (x_val   - x_min) / (x_max - x_min + 1e-8)
x_test  = (x_test  - x_min) / (x_max - x_min + 1e-8)

# ─────────────────────────────────────────────
# CNN architecture
# ─────────────────────────────────────────────
def build_model(input_shape, num_classes):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

model = build_model(input_shape, num_classes)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
checkpoint_path = os.path.join(script_dir, "best_model.keras")

cbs = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy",
                               save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                             restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=5, min_lr=1e-6, verbose=1),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=cbs,
)

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

y_pred       = np.argmax(model.predict(x_test), axis=1)
y_true       = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_classes))

# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "training_curves.png"), dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(max(8, num_classes // 2), max(6, num_classes // 2)))
sns.heatmap(cm, annot=(num_classes <= 20), fmt="d",
            xticklabels=label_classes, yticklabels=label_classes,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "confusion_matrix.png"), dpi=150)
plt.show()

print(f"\nModel saved to: {checkpoint_path}")
