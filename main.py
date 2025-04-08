import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator




images_path = "./images"
labels_path = "./labels"
IMG_SIZE = 224

# for recognition maybe crop the image only to include the tumour for better accuracy in training?(data sugmentation step) 📝
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



# data augmentation step to try to increase accuracy and imprpove generalization 📝
# horrendous results idk try again 
# train_datagen = ImageDataGenerator(
#     rescale=1./255,          
#     rotation_range=30,       
#     width_shift_range=0.2,   
#     height_shift_range=0.2,  
#     shear_range=0.2,          
#     zoom_range=0.2,          
#     horizontal_flip=True,    
#     fill_mode='nearest'      
# )

# val_datagen = ImageDataGenerator(rescale=1./255)


# Lets try taining and testing first without validation and look at the accuracy 📝
# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True,)

# Adding the validation
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

# tried out the 60 20 20 split
# erm kinda bad
# X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, random_state=42, shuffle=True)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)


# checking class imbalance📝
print(np.unique(y_train, return_counts=True))

# might be too complex leading to overfitting 📝
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

# simplified the model to prevent overfitting 📝
# kind of bad though 
# model = Sequential([
#     Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
#     MaxPooling2D(pool_size=(2, 2)),

#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Flatten(),
#     Dense(32, activation='relu'),
#     Dense(1, activation='sigmoid')  
# ])


# adding L2 regularization to help with teh models accuracy and decrease the loss 📝
# em literal garbage accuracy was 0.42 which is worse than without regularizing so forget about this for now
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1), kernel_regularizer=l2(0.01)),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Flatten(),
#     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# history = model.fit(X_train, y_train, epochs=10, batch_size=16)
# fitting including the validation 📝
# history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

# add stop early! 📝
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# history = model.fit(X_train, y_train, epochs=100, batch_size=16, 
#                     validation_data=(X_val, y_val), callbacks=[early_stopping])

# asjusting the learning rate dynamically might help the accuracy 📝
# also experimentred with a different batch size initially was 16 and then increased it to 32📝
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)
history = model.fit(X_train, y_train, epochs=100, batch_size=64, 
                    validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr])


# part of data augmentation step
# history = model.fit(
#     train_datagen.flow(X_train, y_train, batch_size=32),  
#     epochs=100,
#     validation_data=val_datagen.flow(X_val, y_val, batch_size=32),  
#     callbacks=[early_stopping, reduce_lr]
# )


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
