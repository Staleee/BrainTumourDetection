import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



images_path = "./images"
labels_path = "./labels"
IMG_SIZE = 224

# First we need to load the labels and images
# Images are normalizaed to [0,1] scale and read as grayscale
# tumor class is stored in the labels list that will be returned, so bounding boxes are ignored at this stage(unless they want detection)
def load_data(images_path, labels_path):
    images = []
    labels = []
    
    for filename in os.listdir(images_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(images_path, filename)
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            
            label_filename = filename.replace(".jpg", ".txt")
            label_path = os.path.join(labels_path, label_filename)
            
            with open(label_path, "r") as file:
                line = file.readline().strip().split()
                tumor_class = int(line[0])  
                labels.append(tumor_class)  

    return np.array(images), np.array(labels)


load_data(images_path, labels_path)