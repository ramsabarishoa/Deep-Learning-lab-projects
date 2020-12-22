# -*- coding: utf-8 -*-
"""Hyperparameter_Tuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iOwTkoTcTD_zlS23KmtRCEZ09i_65RUl
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import zipfile
import glob, os
import sys
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

print(tf.__version__)

from google.colab import drive
drive.mount('/content/drive')

!unzip -uq "/content/drive/MyDrive/idrid.zip" -d "/content/drive/MyDrive"
print('Unzipped the contents to the drive')

batch_size = 32
img_ht = 256
img_wd = 256

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 600)

np.set_printoptions(threshold=1000)

train_images = glob.glob("/content/drive/MyDrive/IDRID_dataset/images/train/*.jpg")
print('Total number of training images:',len(train_images))

df_train = pd.read_csv('/content/drive/MyDrive/IDRID_dataset/labels/train.csv')
df_train = df_train.drop_duplicates()
df_train = df_train.iloc[:, : 2]
#print(df_train.head())
df_train[['Retinopathy grade']].hist(figsize = (10, 5))
plt.title('Groundtruth Labels for Diabetic Retinopathy before binarization')
plt.xlabel('Label')
plt.ylabel('Number of Images')

df_train['Retinopathy grade'] = df_train['Retinopathy grade'].replace([0,1,2,3,4],[0,0,1,1,1])
df_train[['Retinopathy grade']].hist(figsize = (10, 5))
df_train = df_train.sample(frac=1).reset_index(drop=True)
plt.title('Groundtruth Labels for Diabetic Retinopathy after binarization')
plt.xlabel('Label')
plt.ylabel('Number of Images')

N_Training = round(len(df_train) * 0.8)
train = df_train[:N_Training]
validation = df_train[N_Training:]
print('---------------------------------------------------------')
print('Splitting the train samples into train and validation set')
print('---------------------------------------------------------')
print('Number of training samples:',len(train))
print('Number of validation samples:',len(validation))

label_0 = train[train['Retinopathy grade'] == 0]
label_1 = train[train['Retinopathy grade'] == 1]
print('Label 0:',len(label_0),'\n','Label 1:',len(label_1))

label_count_1, label_count_0 = train['Retinopathy grade'].value_counts()
label_0 = label_0.sample(label_count_1,replace=True)
df_train_sampled = pd.concat([label_0,label_1])
df_train_sampled = df_train_sampled.sample(frac=1,random_state=0)
print(len(label_0),len(label_1))
#print(df_train_sampled)
#df_train_sampled[['Retinopathy grade']].hist(figsize = (10, 5))

df_train_sampled['Image name'] = df_train_sampled['Image name'] + '.jpg'
validation['Image name'] = validation['Image name'] + '.jpg'

train_images_list = []
train_labels_list = []
for tname, tclass in df_train_sampled.itertuples(index=False):
    for ft in train_images:
      if os.path.basename(ft) == tname:
        #print(fp,iname,iclass)
        train_images_list.append(ft)
        train_labels_list.append(tclass)

val_images_list = []
val_labels_list = []
for vname, vclass in validation.itertuples(index=False):
  for fv in train_images:
      if os.path.basename(fv) == vname:
        #print(fv,vname,vclass)
        val_images_list.append(fv)
        val_labels_list.append(vclass)

val_img = np.array([img_to_array(load_img(img, target_size=(256, 256)))for img in val_images_list]).astype('float32')
val_labels_list = np.array(val_labels_list)

print(len(train_images_list),len(train_labels_list))
print(len(val_images_list), len(val_labels_list))

def _parse_function(image, label):
  img_train = tf.io.read_file(image)
  img_decoded = tf.io.decode_jpeg(img_train)
  img_cropped = tf.image.central_crop(img_decoded, central_fraction=0.95)
  img_cropped_bound = tf.image.crop_to_bounding_box(img_cropped, 0 , 0 , target_height = 2700, target_width = 3580)
  image_cast = tf.cast(img_cropped_bound, tf.float32) 
  image_cast = image_cast / 255.0
  image_resized = tf.image.resize(image_cast,size=(img_ht,img_wd))
  return image_resized, label

def build_dataset(images, labels):
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  dataset = dataset.cache()
  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(len(images))
  dataset = dataset.prefetch(AUTOTUNE)
  return dataset

train_dataset = build_dataset(train_images_list, train_labels_list)
#Debug
print(tf.data.experimental.cardinality(train_dataset).numpy())

def to_train_datagen():
  for image, label in train_dataset:
    image_matrix = image.numpy()
    label_matrix = label.numpy()
    print(image_matrix.shape)
    print(label_matrix.shape)

  return image_matrix, label_matrix

train_image_array, train_label_array = to_train_datagen()

# Create train generator.
train_datagen = ImageDataGenerator(rotation_range=30, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(train_image_array, train_label_array, shuffle=False, batch_size=batch_size)

# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = val_datagen.flow(val_img, val_labels_list, shuffle=False, 
                                   batch_size=batch_size)

test_images = glob.glob("/content/drive/MyDrive/IDRID_dataset/images/test/*.jpg")
print(len(test_images))
df_test = pd.read_csv('/content/drive/MyDrive/IDRID_dataset/labels/test.csv')
df_test['Retinopathy grade'] = df_test['Retinopathy grade'].replace([0,1,2,3,4],[0,0,1,1,1])
df_test['Image name'] = df_test['Image name'] + '.jpg'
df_test = df_test.drop_duplicates()
df_test = df_test.iloc[:, : 2]
#print(df_test)

test_images_list = []
test_labels = []
for tname, tclass in df_test.itertuples(index=False):
  for ft in test_images:
    if os.path.basename(ft) == tname:
      #print(ft,tname,tclass)
      t_img = tf.io.read_file(ft)
      t_img_decoded = tf.io.decode_jpeg(t_img)

      t_img_cropped = tf.image.central_crop(t_img_decoded, central_fraction=0.95)
      t_img_cropped_bound = tf.image.crop_to_bounding_box(t_img_cropped, 0 , 0 , target_height = 2700, target_width = 3580)
      
      t_image_cast = tf.cast(t_img_cropped_bound, tf.float32) 
      t_image_cast = t_image_cast / 255.0
      t_image_resized = tf.image.resize(t_image_cast,size=(img_ht,img_wd))
      test_images_list.append(t_image_resized)
      test_labels.append(tclass)

test_images_list = tf.convert_to_tensor(test_images_list)
test_labels = tf.convert_to_tensor(test_labels)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

from tensorboard.plugins.hparams import api as hp

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 512]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2, 0.3, 0.4, 0.5]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','sgd']))
HP_EPOCHS = hp.HParam('epochs',hp.Discrete([100, 200]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
      hparams = [HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_EPOCHS],
      metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_test_model(hparams):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(hparams[HP_NUM_UNITS], kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu))
  model.add(Dropout(hparams[HP_DROPOUT]))
  model.add(Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu))
  model.add(Dropout(hparams[HP_DROPOUT]))
  model.add(Dense(2, activation='softmax'))


  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  model.fit(train_generator, validation_data=val_generator, epochs=hparams[HP_EPOCHS])

  _, accuracy = model.evaluate(test_images_list, test_labels)
  return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in HP_DROPOUT.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
      for epochs in HP_EPOCHS.domain.values:

        hparams = {
            HP_NUM_UNITS: num_units,
            HP_DROPOUT: dropout_rate,
            HP_OPTIMIZER: optimizer,
            HP_EPOCHS: epochs,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1

# Commented out IPython magic to ensure Python compatibility.
# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/hparam_tuning
