import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

images_path = "./images"
labels_path = "./labels"
IMG_SIZE = 224

# for recognition maybe crop the image only to include the tumour for better accuracy in training?(data sugmentation step)
def load_data(images_path, labels_path):
    images = []
    labels = []

    for filename in os.listdir(images_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(images_path, filename)
            label_filename = filename.replace(".jpg", ".txt")
            label_path = os.path.join(labels_path, label_filename)

            if not os.path.exists(label_path):
                print(f"Skipping {filename}: Label file not found")
                continue  

            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
                img_array = img_to_array(img) / 255.0
            except Exception as e:
                print(f"Skipping image {filename}: {e}")
                continue  

            with open(label_path, "r") as file:
                line = file.readline().strip().split()
                try:
                    tumor_class = int(line[0])  
                except ValueError:
                    print(f"Skipping label {label_filename}: Invalid format")
                    continue 

            images.append(img_array)
            labels.append(tumor_class)

    return np.array(images), np.array(labels)

images, labels = load_data(images_path, labels_path)
print(f"Loaded {len(images)} images and {len(labels)} labels")


# Lets try taining and testing first without validation and look at the accuracy
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True,)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=16)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
