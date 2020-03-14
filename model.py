import tensorflow as tf
import numpy as np

obj_dic = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 
  6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 
  12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 
  17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}


class VOC_Model():
  def __init__(self, training_set, validation_set, learning_rate, num_epochs, img_width, img_height):
    self.training_set = training_set
    self.validation_set = validation_set
    self.num_epochs = num_epochs
    self.lr = learning_rate
    self.width = img_width
    self.height = img_height
    self.model = self.get_model()
    self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

  def get_model(self):
    base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(self.width, self.height, 3), pooling=None, classes=20)
    base_model.trainable = False
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(.3)(x)
    output = tf.keras.layers.Dense(4+len(obj_dic))(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    # model.summary()
    return model
  
  def detection_loss(self, model_output, target_bb, target_category, depth):
    model_bb = model_output[:,:4]
    model_class = model_output[:,4:]
    #One hot encode our target that should be just an integer. (We do this cause sparse_categorical_crossentropy isn't working)
    target_category = tf.one_hot(target_category, depth)
    #casting category to one hot float32 because we are expecting probabilistic activation of our classes from model output. 
    target_category = tf.cast(target_category, dtype=tf.float32)
    model_bb = tf.math.sigmoid(model_bb)*224
    target_bb = target_bb*224

    model_class = tf.nn.softmax(model_class)
    mae_loss = tf.keras.losses.MAE(target_bb, model_bb)
    scce_loss = tf.keras.losses.categorical_crossentropy(target_category, model_class)
    return mae_loss + scce_loss

  @tf.function
  def train_model(self, inputs, target_bbs, target_categories):
    with tf.GradientTape() as t:
      outputs = self.model(inputs)
      current_loss = self.detection_loss(outputs, target_bbs, target_categories, len(obj_dic))
    gradients = t.gradient(current_loss, self.model.trainable_variables)
    self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
    return current_loss

  def fit(self):
    for _ in range(self.num_epochs):
      iteration_obj = next(iter(self.training_set))
      train, label, category = iteration_obj
      curr_loss = self.train_model(train, label, category)
      print(curr_loss.numpy().mean())
    self.model.save_weights('resnet50_e30_weights.h5')