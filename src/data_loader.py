import tensorflow as tf
import numpy as np
import os

IMG_SIZE = 224

def parse_label_file(label_path):
    label_path = label_path.decode("utf-8")
    with open(label_path, 'r') as f:
        line = f.readline().strip()
        class_id, cx, cy, w, h = map(float, line.split())
        x1 = (cx - w / 2) * IMG_SIZE
        y1 = (cy - h / 2) * IMG_SIZE
        x2 = (cx + w / 2) * IMG_SIZE
        y2 = (cy + h / 2) * IMG_SIZE
        return np.array([x1, y1, x2, y2, class_id], dtype=np.float32)

def load_image_and_label(image_path, label_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.numpy_function(parse_label_file, [label_path], tf.float32)
    label.set_shape([5])
    return image, label

def create_dataset(image_dir, label_dir):
    image_paths, label_paths, test_images = [], [], []

    for fname in os.listdir(image_dir):
        if fname.endswith('.jpg'):
            img_path = os.path.join(image_dir, fname)
            lbl_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))

            if os.path.exists(lbl_path):
                image_paths.append(img_path)
                label_paths.append(lbl_path)
            else:
                test_images.append(img_path)

    labeled_ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    labeled_ds = labeled_ds.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(test_images)
    test_ds = test_ds.map(lambda x: load_image_and_label(x, x), num_parallel_calls=tf.data.AUTOTUNE)

    return labeled_ds, test_ds
