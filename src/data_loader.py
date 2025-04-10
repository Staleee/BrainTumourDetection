import os
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

def parse_label_file(label_path):
    path = label_path.decode("utf-8")
    with open(path, 'r') as f:
        line = f.readline().strip()
        if not line:
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)
        class_id, cx, cy, w, h = map(float, line.split())
        x1 = (cx - w / 2) * IMG_SIZE
        y1 = (cy - h / 2) * IMG_SIZE
        x2 = (cx + w / 2) * IMG_SIZE
        y2 = (cy + h / 2) * IMG_SIZE
        return np.array([x1, y1, x2, y2, class_id], dtype=np.float32)

def load_image_and_label(image_path, label_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    label = tf.numpy_function(parse_label_file, [label_path], tf.float32)
    label.set_shape([5])
    return image, label

def create_dataset(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    label_paths = [os.path.join(label_dir, f.replace('.jpg', '.txt')) for f in image_files]

    valid_pairs = [(img, lbl) for img, lbl in zip(image_paths, label_paths) if os.path.exists(lbl)]
    image_paths, label_paths = zip(*valid_pairs)

    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(label_paths)))
    dataset = dataset.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
