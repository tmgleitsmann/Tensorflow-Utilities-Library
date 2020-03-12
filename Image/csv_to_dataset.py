import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

#Once object is initialized you can call 

#args: 
# Posix path to CSV file. 
# Batch Size (number) - to create dataset objects.
# Width (number) - to reshape image width.
# Height (number) - to reshape image height.
# Train_test_split (number) - 0.0 - 1.0 of how much of your data you'd like to allocate to training dataset. Rest to validation.
# Image Augmentation Bool (boolean) - True or False if you want your image data augmented. (no zoom augmentation)

###IMPORTANT: If image augmentation is selected, labels will not augment with them.
###           This means for bounding box predictions that this flag should be disabled.

class DataLoader:
  def __init__(self, csv_path, batch_size, width, height, channels = 3, train_test_split = .9, aug_flag = True, labels_to_float=False):
    self.aug_flag = aug_flag
    self.batch_size = batch_size
    self.convert_labels = labels_to_float
    self.width = width
    self.height = height
    self.split = train_test_split
    self.channels = channels
    self.training_dataset, self.validation_dataset = self._csv_to_dataset(csv_path)
  
  def __call__(self):
    return self.training_dataset, self.validation_dataset

  def _rotate_image_aug(self, image):
    degree = random.random()*360
    image = tfa.image.rotate(image, degree*math.pi/180, interpolation='BILINEAR')
    return image

  def _flip_image_aug(self, image):
    image = tf.image.random_flip_left_right(image, seed=None)
    image = tf.image.random_flip_up_down(image, seed=None)
    return image

  def _string_to_float(self, labels_list):
    ret_list = []
    for string_val in labels_list:
      temp_list = string_val.split()
      instance_list = []
      for string_num in temp_list:
        instance_list.append(float(string_num))
      ret_list.append(instance_list)
    return ret_list

  def image_augmentation(self, image):
    image = self._flip_image_aug(image)
    image = self._rotate_image_aug(image)
    return image

  def pre_processing(self, filename, label):
    img_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(img_string, channels=self.channels)
    image_normed = (tf.cast(image_decoded, tf.float32)/255.) #normalize our data between 0 and 1
    image_resized = tf.image.resize(image_normed, (self.width, self.height)) #resize our image to input tensor size
    if self.aug_flag:
      image_resized = self.image_augmentation(image_resized)
    return image_resized, label

  def _csv_to_dataset(self, csv_path):
    data = pd.read_csv(csv_path)
    #grabbing the second and third column of pandas dataframe (Assuming first is numeric order column)
    images_list = data[data.columns[0]].to_list()
    labels_list = data[data.columns[1]].to_list()
    
    #Convert your labels to list of lists
    if self.convert_labels:
      labels_list = self._string_to_float(labels_list)
    
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(
      images_list, labels_list, train_size=self.split, random_state=100)

    train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels))).map(self.pre_processing).shuffle(buffer_size=10000).batch(self.batch_size)
    self.aug_flag = False # we do not want to augment the validation data.
    val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels))).map(self.pre_processing).batch(self.batch_size)
    return train_data, val_data


  # def grab_datasets(self):
  #   return self.training_dataset, self.validation_dataset
