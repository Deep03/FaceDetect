from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, AveragePooling2D, GlobalAveragePooling2D, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers.legacy import Adam
from keras.preprocessing.image import load_img, img_to_array
import inception_mdls
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pathlib


# Input layer
input_layer = Input(shape=(112, 112, 3), name='input_layer')

# Conv1
conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input_layer)

# Max Pool + Norm
max_pool_norm = MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='max_pool_norm')(conv1)

# Inception (2)
inception_2 = inception_mdls.inception_module(max_pool_norm, 64, 64, 192, 1, 1, 1, name='inception_2')


# Norm + Max Pool
norm_max_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='norm_max_pool')(inception_2)

# Inception Layer derived from https://arxiv.org/pdf/1503.03832.pdf & https://arxiv.org/pdf/1409.4842.pdf
inception_3a = inception_mdls.inception_module(norm_max_pool, 64, 96, 128, 16, 32, 32, name='inception_3a')
inception_3b =  inception_mdls.inception_module(inception_3a, 64, 96, 128, 32, 64, 64, name='inception_3b')
inception_3c =  inception_mdls.inception_module(inception_3b, 128, 128, 256, 32, 64, 64, name='inception_3c')

inception_4a =  inception_mdls.inception_module(inception_3c, 256, 96, 192, 32, 64, 128, name='inception_4a')
inception_4b =  inception_mdls.inception_module(inception_4a, 224, 112, 224, 32, 64, 128, name='inception_4b')
inception_4c =  inception_mdls.inception_module(inception_4b, 192, 128, 256, 32, 64, 128, name='inception_4c')
inception_4d =  inception_mdls.inception_module(inception_4c, 160, 144, 288, 32, 64, 128, name='inception_4d')
inception_4e =  inception_mdls.inception_module(inception_4d, 160, 160, 256, 64, 128, 128, name='inception_4e')


inception_5a =  inception_mdls.inception_module(inception_4e, 384, 192, 384, 48, 128, 128, name='inception_5a')
inception_5b =  inception_mdls.inception_module(inception_5a, 384, 192, 384, 48, 128, 128, name='inception_5b')

avgpool = AveragePooling2D()(inception_5b)

fully_conn = Dense(128, activation='relu', name='fully_conn')(avgpool)

fully_conn = Dense(2000)(fully_conn)

global_avg_pool = GlobalAveragePooling2D()(fully_conn)

l2_norm = BatchNormalization()(global_avg_pool)

# FIX Needed: use layers.concatenate
model = Model(inputs=input_layer, outputs=l2_norm)

# lr = 10^-3 described in the paper
# loss(TBC)=triplet_loss
model.compile(Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


data_dir = 'test/'
data_dir = pathlib.Path(data_dir).with_suffix('')

person1 = list(data_dir.glob('0/*'))

# open image using PIL
# Image.open(str(person1[0])).show()


batch_size = 32
img_height = 112
img_width = 112

# dataset to train
train_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# dataset to validate model
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE


train_ds = train_ds.shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache()
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache()
image_batch, labels_batch = next(iter(normalized_train_ds))
first_image = image_batch[0]


# 50 for optimized
epochs=50
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


# stat visualization
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()