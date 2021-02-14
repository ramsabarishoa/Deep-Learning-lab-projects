'''Metrics : Confusion Matrix
Displays the true postives and true negatives'''

import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from input_pipeline.datasets import load
import numpy as np
from evaluation.eval import mdl
import matplotlib.pyplot as plt
from input_pipeline.preprocessing import img_width,img_height
from input_pipeline.datasets import dataset_path
import seaborn as sb
import os
import glob
import pandas as pd

"""Metrics : Confusion Matrix to predict the model test results"""

#The test images are loaded along with their ground truths
test_images = glob.glob(dataset_path + "/images/test/*.jpg")
print('Total number of test images:', len(test_images))
df_test = pd.read_csv(dataset_path + '/labels/test.csv')
df_test['Retinopathy grade'] = df_test['Retinopathy grade'].replace([0, 1, 2, 3, 4], [0, 0, 1, 1, 1])
df_test['Image name'] = df_test['Image name'] + '.jpg'
df_test = df_test.drop_duplicates()
df_test = df_test.iloc[:, : 2]
#Debug
#print(df_test)

test_images_list = []
test_labels = []
predicted_label_list = []
for tname, tclass in df_test.itertuples(index=False):
  for ft in test_images:
    if os.path.basename(ft) == tname:
      #print(ft,tname,tclass)
      t_img = tf.io.read_file(ft) #Read the test image
      t_img_decoded = tf.io.decode_jpeg(t_img)

      t_img_cropped = tf.image.central_crop(t_img_decoded, central_fraction=0.95)
      t_img_cropped_bound = tf.image.crop_to_bounding_box(t_img_cropped, 0 , 0 , target_height = 2700, target_width = 3580)
      
      t_image_cast = tf.cast(t_img_cropped_bound, tf.float32) 
      t_image_cast = t_image_cast / 255.0
      t_image_resized = tf.image.resize(t_image_cast,size=(img_height,img_width))
      t_image_reshape = tf.reshape(t_image_resized, [1,256,256,3])
      x = mdl.predict(t_image_resized) #Predict the label using the compiled model
      predicted_label = np.argmax(x)
      predicted_label_list.append(predicted_label) #Append all the predicted labels to the list
    
      test_images_list.append(t_image_resized)
      test_labels.append(tclass)

test_images_list = tf.convert_to_tensor(test_images_list)
test_labels = tf.convert_to_tensor(test_labels)

df_test['Predicted Class'] = predicted_label_list
df_test['Result'] = np.where(df_test['Retinopathy grade'] == df_test['Predicted Class'], 'Correct Prediction', 'Incorrect Prediction')
#print(df_test)
df_test['Result'].value_counts() #The correct and incorrect labels are listed
cm = confusion_matrix(df_test['Retinopathy grade'],df_test['Predicted Class']) #Plot the confusion matrix
plt.figure(figsize = (10,5))
plt.title('Confusion Matrix')
sb.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
print('Test Accuracy:',metrics.accuracy_score(df_test['Retinopathy grade'], df_test['Predicted Class'])) #Obtain the test accuracy
