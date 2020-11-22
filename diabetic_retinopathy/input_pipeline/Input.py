import tensorflow as tf
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
print(tf.__version__)

#Appending an extension for the first column which has the image name
def append_ext(fn):
    return fn + ".jpg"

df_train_csv = pd.read_csv('/home/swetha/IDRID_dataset/labels/train.csv',dtype=str)
df_train_csv["Image name"]=df_train_csv["Image name"].apply(append_ext)
df_test_csv = pd.read_csv('/home/swetha/IDRID_dataset/labels/test.csv',dtype=str)
df_test_csv["Image name"]=df_test_csv["Image name"].apply(append_ext)

train_folder = '/home/swetha/IDRID_dataset/images/train'
test_folder = '/home/swetha/IDRID_dataset/images/test'

datagen = ImageDataGenerator(validation_split=0.2)

'''Training Dataset'''
train_data = datagen.flow_from_dataframe(dataframe= df_train_csv,
directory = train_folder, x_col = "Image name",
y_col = "Retinopathy grade", subset="training", batch_size=32, seed = 42,class_mode='categorical',shuffle = True)

'''Validation Dataset'''
validation_data = datagen.flow_from_dataframe(dataframe= df_train_csv,
directory = train_folder, x_col = "Image name",
y_col = "Retinopathy grade", subset="validation", seed = 42,class_mode='categorical',shuffle = True)

'''Test Dataset'''
test_data = datagen.flow_from_dataframe(dataframe= df_test_csv,
directory = test_folder, x_col = "Image name",
y_col = "Retinopathy grade", seed = 42,class_mode='categorical')

images_train, labels_train = next(iter(train_data))
images_test, labels_test = next(iter(test_data))
images_validation, labels_validation = next(iter(validation_data))

ds_train = tf.data.Dataset.from_generator(
    lambda: datagen.flow_from_dataframe(train_data),
    output_types=(tf.string, tf.int8),
    output_shapes=(images_train.shape,labels_train.shape)
)
ds_validation = tf.data.Dataset.from_generator(
    lambda: datagen.flow_from_dataframe(validation_data),
    output_types=(tf.string, tf.int8),
    output_shapes=(images_validation.shape,labels_validation.shape)
)

ds_test =tf.data.Dataset.from_generator(
    lambda: datagen.flow_from_dataframe(test_data),
    output_types=(tf.string, tf.int8),
    output_shapes=(images_test.shape,labels_test.shape)
)

print(ds_train.element_spec)
print(ds_validation.element_spec)
print(ds_test.element_spec)

arr = np.ndarray(ds_train)
arr_ = np.squeeze(arr)
plt.imshow(arr_)
plt.show()