import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from input_pipeline.datasets import load
import matplotlib.pyplot as plt
batch_size = 32
img_height = 256
img_width = 256

print('Tensorflow version used here is : ', tf.__version__)
def preprocess(image, label):
  """Dataset preprocessing: Normalizing and resizing"""
  img_train = tf.io.read_file(image)
  img_decoded = tf.io.decode_jpeg(img_train)
  img_cropped = tf.image.central_crop(img_decoded, central_fraction=0.95)
  img_cropped_bound = tf.image.crop_to_bounding_box(img_cropped, 0 , 0 , target_height = 2700, target_width = 3580)
  image_cast = tf.cast(img_cropped_bound, tf.float32)
  image_cast = image_cast / 255.0
  image_resized = tf.image.resize(image_cast,size=(img_height,img_width))
  return image_resized, label

def build_dataset(images, labels):
  """Creating an efficient input pipeline in tensorflow using tf.data.Dataset"""
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  dataset = dataset.cache()
  dataset = dataset.map(preprocess)
  dataset = dataset.batch(len(images))
  dataset = dataset.prefetch(AUTOTUNE)
  return dataset

train_images_list, train_labels_list, val_images_list, val_labels_list = load.to_preprocessing()
val_img = np.array([img_to_array(load_img(img, target_size=(img_height, img_width)))for img in val_images_list]).astype('float32')
val_labels_list = np.array(val_labels_list)
train_dataset = build_dataset(train_images_list, train_labels_list)
#Debug
#print(tf.data.experimental.cardinality(train_dataset).numpy())

def to_train_datagen():
  for image, label in train_dataset:
    image_array = image.numpy()
    label_array = label.numpy()
    print('Shape of the images:', image_array.shape)
    print('Shape of the label array:', label_array.shape)

  return image_array, label_array

train_image_array, train_label_array = to_train_datagen()

# Create train generator.
train_datagen = ImageDataGenerator(rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(train_image_array, train_label_array,
                                     shuffle=False, batch_size=batch_size)

# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = val_datagen.flow(val_img, val_labels_list, shuffle=False,
                                   batch_size=batch_size)



debug_mode_input_pipeline_processing = int(input('Enter 1 for enabling debug option(DEBUG_INPUT_PIPELINE_PREPROCESSING) else enter 0 :'))
if debug_mode_input_pipeline_processing == 1:
  test_image_path = input('Please enter the path for test image:')
  test_img = tf.io.read_file(test_image_path)
  test_img_decoded = tf.io.decode_jpeg(test_img)
  test_img_cropped = tf.image.central_crop(test_img_decoded, central_fraction=0.95)
  test_img_cropped_bound = tf.image.crop_to_bounding_box(test_img_cropped, 0, 0, target_height=2700, target_width=3580)
  test_image_cast = tf.cast(test_img_cropped_bound, tf.float32)
  test_image_cast = test_image_cast / 255.0
  test_image_resized = tf.image.resize(test_image_cast, size=(img_height, img_width))
  plt.imshow(test_image_resized)
  plt.title('Test Image after decoding and resizing')
  plt.show()

  aug = [next(train_generator) for i in range(0, 5)]
  fig, ax = plt.subplots(1, 5, figsize=(15, 6))
  print('Labels:', [item[1][0] for item in aug])
  l = [ax[i].imshow(aug[i][0][0]) for i in range(0, 5)]
  plt.title('Sample images from the train data after augmentation')
  plt.show()
