# -*- coding: utf-8 -*-
"""drd(75) with visualisation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12_C7YePcPE4KNsSZJLBnkNRV8nPmGFfq
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import zipfile
import glob, os
import sys
import matplotlib.pyplot as plt
import tensorflow.keras
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

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', 
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

logdir = "/content/drive/MyDrive/Colab Notebooks/Logdir"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_dir = '/content/drive/MyDrive/Colab Notebooks/checkpoints/' + 'epochs:{epoch:03d}-val_accuracy:{val_accuracy:.3f}.hdf5'
checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                       monitor='val_accuracy',
                                                       verbose=1,

                                                       save_best_only=False,
                                                       mode='max',save_freq="epoch")
model.summary()

call_bks = [tensorboard_callback,checkpoint_callbk]
epochs=120
history = model.fit(
  train_generator,
  validation_data=val_generator,
  epochs=epochs
  
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(18, 9))
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

#Prediction
#Loading test images 

test_images = glob.glob("/content/drive/MyDrive/IDRID_dataset/images/test/*.jpg")
#print(len(test_images))
df_test = pd.read_csv('/content/drive/MyDrive/IDRID_dataset/labels/test.csv')
df_test['Retinopathy grade'] = df_test['Retinopathy grade'].replace([0,1,2,3,4],[0,0,1,1,1])
df_test['Image name'] = df_test['Image name'] + '.jpg'
df_test = df_test.drop_duplicates()
df_test = df_test.iloc[:, : 2]
#print(df_test)
predicted_label_list = []
for iname,iclass in df_test.itertuples(index=False):
    for file in test_images:
      if os.path.basename(file) == iname:
        img = tf.io.read_file(file)
        img = tf.io.decode_jpeg(img)
        img = tf.cast(img,tf.float32) / 255
        img = tf.image.resize_with_pad(img,img_ht,img_wd,antialias=True)
        img = tf.reshape(img, [1,256,256,3])
        x = model.predict(img)
        predicted_label = np.argmax(x)
        predicted_label_list.append(predicted_label)

df_test['Predicted Class'] = predicted_label_list
df_test['Result'] = np.where(df_test['Retinopathy grade'] == df_test['Predicted Class'], 'Correct Prediction', 'Incorrect Prediction')
#print(df_test)

df_test['Result'].value_counts()

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sb
cm = confusion_matrix(df_test['Retinopathy grade'],df_test['Predicted Class'])
plt.figure(figsize = (10,5))
plt.title('Confusion Matrix')
sb.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
print('Test Accuracy:',metrics.accuracy_score(df_test['Retinopathy grade'], df_test['Predicted Class']))

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

from tensorflow import keras

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

img_size = (256, 256)
last_conv_layer_name = "conv2d_3"
classifier_layer_names = ["max_pooling2d_3","dropout","flatten","dense","dropout_1",
    "dense_1"
]

# The local path to our target image

def get_img_array(img_path):
    '''# `img` is a PIL image of size 256x256
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (256, 256, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 256, 256, 3)'''
    img_train = tf.io.read_file(img_path)
    img_decoded = tf.io.decode_jpeg(img_train)
    img_cropped = tf.image.central_crop(img_decoded, central_fraction=0.95)
    img_cropped_bound = tf.image.crop_to_bounding_box(img_cropped, 0 , 0 , target_height = 2700, target_width = 3580)
    image_cast = tf.cast(img_cropped_bound, tf.float32) 
    image_cast = image_cast / 255.0
    array= tf.image.resize(image_cast,size=(img_ht,img_wd))
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

img_array = get_img_array("/content/drive/MyDrive/IDRID_dataset/images/test/IDRiD_035.jpg")




# Print what the top predicted class is
preds = model.predict(img_array)
#print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)

# Display heatmap
plt.matshow(heatmap)
plt.show()

# We load the original image
img = keras.preprocessing.image.load_img("/content/drive/MyDrive/IDRID_dataset/images/test/IDRiD_035.jpg")
img =img.resize((256,256))
img = keras.preprocessing.image.img_to_array(img)
#img_array=tf.squeeze(img_array,axis=0)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)
# We use jet colormap to colorize heatmap
jet = cm.get_cmap("viridis")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((256,256))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap*0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = "graded.jpg"
superimposed_img.save(save_path)
img=keras.preprocessing.image.array_to_img(img)
original="img.jpg"
img.save(original)
# Display Grad CAM
display(Image(save_path))
display(Image(original))
print(predicted_label_list[34])
