import tensorflow as tf
import keras 
from keras.models import load_model
from deeplab import DeepLabV3Plus
import matplotlib.pyplot as plt

import pickle
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


import cv2
import numpy as np
import matplotlib.pyplot as plt


H, W = 512, 512
num_classes = 34


image_path = 'test.jpg'
img = tf.io.read_file(image_path)
img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
img = tf.image.resize(images=img, size=[H, W])
img = np.array(img)
img = img.reshape((-1, H, W, 3))

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = DeepLabV3Plus(H, W, num_classes)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

model.load_weights('trained_segmentation.h5')

preds = model.predict(img)

with open('cityscapes_dict.pkl', 'rb') as f:
    id_to_color = pickle.load(f)['color_map']

image = load_img('test.jpg')
image = img_to_array(image)

alpha = 0.5
dims = image.shape
image = cv2.resize(image, (W, H))

z = np.squeeze(preds)
y = np.argmax(z, axis=2)

img_color = image.copy()
for i in np.unique(y):
    if i in id_to_color:
        img_color[y == i] = id_to_color[i]
disp = img_color.copy()

cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)

cv2.imwrite('segmented.jpg', cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))










