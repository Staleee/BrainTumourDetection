import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 224

def load_data(images_path, labels_path):
    images, labels_cls, labels_bbox = [], [], []

    for filename in os.listdir(images_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(images_path, filename)
            label_path = os.path.join(labels_path, filename.replace(".jpg", ".txt"))

            if not os.path.exists(label_path): continue

            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0

            with open(label_path, "r") as file:
                values = list(map(float, file.readline().strip().split()))
                cls = int(values[0])
                bbox = values[1:]  # center_x, center_y, w, h

            images.append(img)
            labels_cls.append(cls)
            labels_bbox.append(bbox)

    return np.array(images), np.array(labels_cls), np.array(labels_bbox)

# Load your data
images, class_labels, bbox_labels = load_data("data/images", "data/labels")

X_train, X_val, y_train_cls, y_val_cls, y_train_bbox, y_val_bbox = train_test_split(
    images, class_labels, bbox_labels, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='linear')  # 1 for class, 4 for bbox
])


def custom_loss(y_true, y_pred):
    cls_true = y_true[:, 0]
    bbox_true = y_true[:, 1:]

    cls_pred = tf.keras.activations.sigmoid(y_pred[:, 0])
    bbox_pred = y_pred[:, 1:]

    cls_loss = tf.keras.losses.binary_crossentropy(cls_true, cls_pred)
    bbox_loss = tf.reduce_mean(tf.abs(bbox_true - bbox_pred), axis=1)

    return cls_loss + 0.05 * bbox_loss

model.compile(optimizer=Adam(1e-4), loss=custom_loss, metrics=['accuracy'])

# Pack both outputs together
y_train = np.concatenate([y_train_cls.reshape(-1,1), y_train_bbox], axis=1)
y_val = np.concatenate([y_val_cls.reshape(-1,1), y_val_bbox], axis=1)

early_stop = EarlyStopping(patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.7, min_lr=1e-6)


history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=32, callbacks=[early_stop, reduce_lr])

# Accuracy
preds = model.predict(X_val)
cls_preds = tf.round(tf.keras.activations.sigmoid(preds[:, 0])).numpy().flatten()
val_cls_true = y_val_cls.flatten()
acc = np.mean(cls_preds == val_cls_true)
print(f"\nValidation Classification Accuracy: {acc:.2f}")

# MAE for BBoxes
mae = np.mean(np.abs(preds[:, 1:] - y_val_bbox))
print(f"Validation Bounding Box MAE: {mae:.2f}")

# Plot Accuracy
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.legend()
plt.title("Classification Accuracy")
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label="Train Loss")
plt.title("Loss")
plt.legend()
plt.show()

# Save model
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/dual_head_model.keras")
model.save_weights("saved_models/dual_head_model_weights.h5")
