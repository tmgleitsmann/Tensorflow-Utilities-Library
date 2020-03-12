import matplotlib.pyplot as plt
import pandas as pd
import re
import tensorflow as tf
import numpy as np

#Utility Functions that might be useful

### scale_bounding_boxes: box coordinates will scale between 0 & 1. Useful if you're normalizing your pixel values between 0 & 1
### denorm_numpy_image: return your image tensor back to pixel values between 0 & 255
### denorm_numpy_labels: return your labels back to scale of 0-255. Scaled by width & height args.


def scale_bounding_boxes(csv_path):
  data = pd.read_csv(csv_path)
  images_list = data[data.columns[0]].to_list()
  labels_list = data[data.columns[1]].to_list()
  scaled_vals = []
  for idx, images in enumerate(images_list):
    height, width, _ = plt.imread(images).shape
    h1, w1, h2, w2 = re.split(r'[;,\s]\s*', labels_list[idx])
    scaled_h1, scaled_h2 = float(h1)/float(height), float(h2)/float(height)
    scaled_w1, scaled_w2 = float(w1)/float(width), float(w2)/float(width)
    to_append = str(round(scaled_h1, 2))+' '+str(round(scaled_w1, 2))+' '+str(round(scaled_h2, 2))+' '+str(round(scaled_w2, 2))
    scaled_vals.append(to_append)

  data[data.columns[1]] = scaled_vals
  data.to_csv(csv_path, index=False)
  return csv_path

def denorm_numpy_image(image):
  return tf.cast(tf.math.scalar_mul(255., image), dtype=tf.uint8)

def denorm_numpy_labels(labels, width, height):
  h1 = labels[0] * height
  h2 = labels[2] * height
  w1 = labels[1] * width
  w2 = labels[3] * width
  return tf.cast(tf.constant(np.array([h1, w1, h2, w2])), dtype=tf.uint8)