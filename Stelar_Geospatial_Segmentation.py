import os
import cv2
from PIL import Image 
import numpy as np 
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from matplotlib import pyplot as plt
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from tensorflow.keras.utils import to_categorical 


dataset_size = 8000

print("data loading started")

#image_dataset = np.load('./dataset/saved_images_masks_npy/images/imagesChunk3.npy')
image_dataset = np.load('./dataset/saved_images_masks_npy/images/testImgs.npy')

#np.save('./dataset/saved_images_masks_npy/images/testImgs.npy', image_dataset)

image_dataset[image_dataset<0] = 0

#labels = np.load('./dataset/saved_images_masks_npy/masks/masksChunk3.npy')
labels = np.load('./dataset/saved_images_masks_npy/masks/testlables.npy')


#np.save('./dataset/saved_images_masks_npy/masks/testlables.npy', test_lables)

print("image_dataset.max()", image_dataset.max())
print("image_dataset.min()", image_dataset.min())


labels = labels.astype(np.int8) 
image_dataset = image_dataset.astype(np.int8) 

torch.cuda.empty_cache()

image_dataset = torch.tensor(image_dataset).to(device)
labels = torch.tensor(labels).to(device)

print("device", device)

print("data loading ended")

image_dataset = image_dataset[:10]
labels = labels[:10]

patch_len_x = 180
patch_len_y = 180

image_splits = []
label_splits = []

for record in range(image_dataset.shape[0]):
  print("record start", record)
  for ind_i in range(int(image_dataset.shape[1]/patch_len_x)):
    for ind_j in range(int(image_dataset.shape[2]/patch_len_y)):
            selected_patch = image_dataset[record][(ind_i * patch_len_x):((ind_i + 1) * patch_len_x), (ind_j * patch_len_y):((ind_j + 1) * patch_len_y)]
            selected_mask = labels[record][(ind_i * patch_len_x):((ind_i + 1) * patch_len_x), (ind_j * patch_len_y):((ind_j + 1) * patch_len_y)]
            image_splits.append(selected_patch)
            label_splits.append(selected_mask)

'''image_splits = np.concatenate(image_splits, axis=0)
label_splits = np.concatenate(label_splits, axis=0)'''

image_splits = torch.cat(image_splits, axis=0).to(device)
label_splits = torch.cat(label_splits, axis=0).to(device)

print("record end", record)


image_splits = image_splits.reshape(-1, patch_len_x, patch_len_y)
label_splits = label_splits.reshape(-1, patch_len_x, patch_len_y)

print("image_splits.shape", image_splits.shape)
print("label_splits.shape", label_splits.shape)


print("categorical started")

total_classes = 10
#labels_categorical_dataset = to_categorical(label_splits.cpu(), num_classes=total_classes)

'''labels_categorical_dataset = []
for i in range(0, 1000, 200):
  labels_categ = to_categorical(label_splits[i:i+200].cpu(), num_classes=total_classes)
  print("labels_categ.shape", labels_categ.shape)
  labels_categorical_dataset.append(labels_categ) 
  print("i", i)
labels_categorical_dataset = torch.tensor(labels_categorical_dataset)
labels_categorical_dataset = torch.cat((labels_categorical_dataset,), axis=0).to(device)
labels_categorical_dataset = labels_categorical_dataset.reshape(-1, patch_len_x, patch_len_y, total_classes)
print("labels_categorical_dataset.shape", labels_categorical_dataset.shape)'''


import torch.nn.functional as F

# Assuming labels is a tensor of integer labels
#label_splits = label_splits.to(torch.int8)

labels_categorical_dataset = torch.tensor([]).to(device)
for i in range(dataset_size//200):
  print("i*200", i*200)
  #label_splits = label_splits.to(torch.int8)
  labels_cat = F.one_hot(label_splits[(i*200):(i*200)+200].long(), num_classes=total_classes)
  labels_categorical_dataset = torch.cat((labels_categorical_dataset, labels_cat), axis=0).to(device)  

image_splits = image_splits[:dataset_size]
print("labels_categorical_dataset.shape merged", labels_categorical_dataset.shape)

'''catg_labels = torch.zeros([label_splits.shape[0], label_splits.shape[1], label_splits.shape[2], 10]).to(device)
catg_labels[:,:,:, label_splits[:,:,:]] = 1'''


#print("catg_labels.shape", catg_labels.shape)

#print("catg_labels", catg_labels)

#

# Convert labels to lower-precision data type (e.g., torch.int8)
#labels = labels.to(torch.int8)

# Perform one-hot encoding on the labels
#labels_categorical_dataset = torch.eye(total_classes, dtype=torch.int8)[labels.long()]


print("labels_categorical_dataset.shape merged", labels_categorical_dataset.shape)

from sklearn.model_selection import train_test_split
master_trianing_dataset = image_splits
labels_categorical_dataset = labels_categorical_dataset
X_train, X_test, y_train, y_test = train_test_split(master_trianing_dataset, labels_categorical_dataset, test_size=0.15, random_state=100)

X_train = X_train.reshape(-1, patch_len_x, patch_len_y, 1)
X_test = X_test.reshape(-1, patch_len_x, patch_len_y, 1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

image_height = X_train.shape[1]
image_width = X_train.shape[2]
image_channels = X_train.shape[3]
total_classes = y_train.shape[3]

print(image_height)
print(image_width)
print(image_channels)
print(total_classes)

##################################################
import tensorflow as tf
X_train = X_train.cpu().numpy()
y_train = y_train.cpu().numpy()
X_test = X_test.cpu().numpy()
y_test = y_test.cpu().numpy()

# Convert NumPy arrays to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
##################################################

# Deep learning part

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, BatchNormalization, Dropout, Lambda
     
from keras import backend as K

def jaccard_coef(y_true, y_pred):
  y_true_flatten = K.flatten(y_true)
  y_pred_flatten = K.flatten(y_pred)
  intersection = K.sum(y_true_flatten * y_pred_flatten)
  final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
  return final_coef_value

def multi_unet_model(n_classes=5, image_height=256, image_width=256, image_channels=1):

  inputs = Input((image_height, image_width, image_channels))

  source_input = inputs

  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(source_input)
  print("c1.shape", c1.shape)
  c1 = Dropout(0.2)(c1)
  print("after drop out c1.shape", c1.shape)
  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
  print("after another convol", c1.shape)
  p1 = MaxPooling2D((2,2))(c1)
  print("after max pooling", p1.shape)

  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
  c2 = Dropout(0.2)(c2)
  print("c2.shape", c2.shape)
  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
  p2 = MaxPooling2D((2,2))(c2)
  print("p2.shape", p2.shape)

  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
  c3 = Dropout(0.2)(c3)
  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
  print("c3.shape", c3.shape)
  p3 = MaxPooling2D((2,2))(c3)
  print("p3.shape", p3.shape)

  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
  c4 = Dropout(0.2)(c4)
  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
  print("c4.shape", c4.shape)
  p4 = MaxPooling2D((2,2))(c4)
  print("p4.shape", p4.shape)

  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
  c5 = Dropout(0.2)(c5)
  print("c5.shape", c5.shape)
  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
  print("again c5.shape", c5.shape)

  u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
  print("u6.shape", u6.shape)
  u6 = concatenate([u6, c4])
  print("after concatenation u6.shape", u6.shape)
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)
  print("c6.shape", c6.shape)
  
  u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
  #u7 = Conv2DTranspose(64, (2, 2), strides=(1, 1), padding="valid")(c6)
  print("u7.shape", u7.shape)
  print("c3.shape", c3.shape)
  u7 = Conv2DTranspose(64, (2,2), strides=(1,1), padding="valid")(u7)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

  u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
  c8 = Dropout(0.2)(c8)
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

  u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
  c9 = Dropout(0.2)(c9)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

  outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)

  model = Model(inputs=[inputs], outputs=[outputs])
  return model





metrics = ["accuracy", jaccard_coef]

def get_deep_learning_model():
  return multi_unet_model(n_classes=total_classes, 
                          image_height=image_height, 
                          image_width=image_width, 
                          image_channels=image_channels)

model = get_deep_learning_model()


weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm

dice_loss = sm.losses.DiceLoss(class_weights = weights)

focal_loss = sm.losses.CategoricalFocalLoss()

dice_loss = sm.losses.DiceLoss(class_weights = weights)

total_loss = dice_loss + (1 * focal_loss)

import tensorflow as tf

tf.keras.backend.clear_session()

model.compile(optimizer="adam", loss=total_loss, metrics=metrics)

from keras.utils.vis_utils import plot_model

plot_model(model, to_file='satellite_model_plot.png', show_shapes=True, show_layer_names=True)

import keras
from IPython.display import clear_output

# To get two plots
class PlotLossEx(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.jaccard_coef = []
        self.val_jaccard_coef = []
        self.fig = plt.figure()
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))

        self.jaccard_coef.append(logs.get("jaccard_coef"))
        self.val_jaccard_coef.append(logs.get("val_jaccard_coef"))

        self.i += 1

        plt.figure(figsize=(14, 8))
        f, (graph1, graph2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)
        
        graph1.set_yscale('log')
        graph1.plot(self.x, self.losses, label="loss")
        graph1.plot(self.x, self.val_losses, label="val_loss")
        graph1.legend()

        graph2.set_yscale('log')
        graph2.plot(self.x, self.jaccard_coef, label="jaccard_coef")
        graph2.plot(self.x, self.val_jaccard_coef, label="val_jaccard_coef")

        graph2.legend()
        plt.show();
    
plot_loss = PlotLossEx()



#Two graph
model_history = model.fit(X_train, y_train,
                          batch_size=16,
                          verbose=1,
                          epochs=10,
                          validation_data=(X_test, y_test),
                          callbacks=[plot_loss],
                          shuffle=False)

history_a = model_history

history_a.history

