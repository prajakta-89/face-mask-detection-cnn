
# ============================================================
# FACE MASK DETECTION USING CNN
# Classes: Incorrect Mask, With Mask, Without Mask
# ============================================================

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/validation")
TEST_DIR = Path("data/test")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# ============================================================
# 2. CHECK DATASET FOLDERS
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder.resolve()}"
        )

print("\nDataset folders verified.")

# ============================================================
# 3. LOAD IMAGES
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(VAL_DIR),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClass names:", class_names)
print("Number of classes:", num_classes)

if num_classes != 3:
    raise ValueError(
        f"Expected 3 classes, but found {num_classes}. "
        "Check your dataset folders."
    )

if (
    val_ds.class_names != class_names
    or test_ds.class_names != class_names
):
    raise ValueError(
        "Training, validation and test class names/order differ."
    )

# Save class order for later predictions
with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

# ============================================================
# 4. PREPROCESS IMAGES
# ============================================================

def normalize(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


train_ds = train_ds.map(
    normalize,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    normalize,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    normalize,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print("\nImage preprocessing completed.")

# ============================================================
# 5. BUILD CNN MODEL
# ============================================================

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Convolutional block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolutional block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolutional block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Feature extraction and classification
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    # Three output classes
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN architecture:")
model.summary()

# ============================================================
# 6. CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_mask_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nStarting CNN training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model with the best weights restored
model.save(str(MODEL_DIR / "face_mask_cnn.keras"))

print("\nTraining completed.")
print("Model saved in:", MODEL_DIR.resolve())

# ============================================================
# 8. PLOT TRAINING ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy_plot.png",
    dpi=300
)
plt.show()

# ============================================================
# 9. PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_plot.png",
    dpi=300
)
plt.show()

# ============================================================
# 10. EVALUATE ON TEST DATA
# ============================================================

print("\nEvaluating on unseen test images...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    cmap="Blues",
    xticks_rotation=30
)

plt.title("Face Mask Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)
plt.show()

# ============================================================
# 13. SAVE TEST RESULTS
# ============================================================

results = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "class_names": class_names,
    "test_images": int(len(y_true))
}

with open(
    RESULTS_DIR / "test_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

print("\nAll tasks completed successfully!")
print("Saved files:")
print("- models/best_mask_model.keras")
print("- models/face_mask_cnn.keras")
print("- models/class_names.json")
print("- results/accuracy_plot.png")
print("- results/loss_plot.png")
print("- results/confusion_matrix.png")
print("- results/classification_report.txt")
print("- results/test_results.json")
