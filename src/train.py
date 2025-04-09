import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import build_model
from data_loader import create_dataset

# === Load Dataset ===
image_dir = "data/images"
label_dir = "data/labels"

full_ds, test_unlabeled = create_dataset(image_dir, label_dir)
full_ds = full_ds.batch(1)

images, labels = [], []
for img, lbl in full_ds:
    images.append(img[0])
    labels.append(lbl[0])

images = np.array(images)
labels = np.array(labels)

X = images
y_bbox = labels[:, :4]
y_class = labels[:, 4:]

# === Train/Val/Test Split ===
X_train, X_temp, y_bbox_train, y_bbox_temp, y_class_train, y_class_temp = train_test_split(
    X, y_bbox, y_class, test_size=0.2, random_state=42
)
X_val, X_test, y_bbox_val, y_bbox_test, y_class_val, y_class_test = train_test_split(
    X_temp, y_bbox_temp, y_class_temp, test_size=0.5, random_state=42
)

def pack_data(x, bbox, cls):
    return x, {"bbox_output": bbox, "class_output": cls}

train_ds = tf.data.Dataset.from_tensor_slices(pack_data(X_train, y_bbox_train, y_class_train)).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices(pack_data(X_val, y_bbox_val, y_class_val)).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices(pack_data(X_test, y_bbox_test, y_class_test)).batch(32)

# === Build Model ===
model = build_model(input_shape=(224, 224, 1))
model.compile(
    optimizer='adam',
    loss={
        "bbox_output": "mse",
        "class_output": "binary_crossentropy"
    },
    metrics={
        "bbox_output": "mae",
        "class_output": "accuracy"
    }
)

# === Train ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
)

# === Evaluate ===
results = model.evaluate(test_ds)
print(f"\n--- Evaluation ---")
print(f"Classification Accuracy: {results[4]*100:.2f}%")
print(f"Bounding Box MAE: {results[2]*224:.2f} pixels")  # convert to pixels

# === Plot Accuracy ===
plt.plot(history.history['class_output_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_class_output_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy')
plt.legend()
plt.grid(True)
plt.show()
