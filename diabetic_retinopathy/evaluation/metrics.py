import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from input_pipeline.datasets import load
import numpy as np
from evaluation.eval import mdl
import matplotlib.pyplot as plt
from input_pipeline.preprocessing import img_width,img_height
import seaborn as sb
import os
import glob
import pandas as pd

"""Metrics : Confusion Matrix to predict the model test results"""

test_images = glob.glob("/home/swetha/IDRID_dataset/images/images/test/*.jpg")
print('Total number of training images:', len(test_images))
df_test = pd.read_csv('/home/swetha/IDRID_dataset/labels/test.csv')
df_test['Retinopathy grade'] = df_test['Retinopathy grade'].replace([0, 1, 2, 3, 4], [0, 0, 1, 1, 1])
df_test['Image name'] = df_test['Image name'] + '.jpg'
df_test = df_test.drop_duplicates()
df_test = df_test.iloc[:, : 2]
# print(df_test)

predicted_label_list = []
for iname, iclass in df_test.itertuples(index=False):
    for file in test_images:
      if os.path.basename(file) == iname:
        img = tf.io.read_file(file)
        img = tf.io.decode_jpeg(img)
        img = tf.cast(img,tf.float32) / 255
        img = tf.image.resize_with_pad(img, img_height, img_width, antialias=True)
        img = tf.reshape(img, [1, img_height, img_width, 3])
        x = mdl.predict(img)
        predicted_label = np.argmax(x)
        predicted_label_list.append(predicted_label)

df_test['Predicted Class'] = predicted_label_list
df_test['Result'] = np.where(df_test['Retinopathy grade'] == df_test['Predicted Class'], 'Correct Prediction', 'Incorrect Prediction')
#print(df_test)
df_test['Result'].value_counts()
cm = confusion_matrix(df_test['Retinopathy grade'],df_test['Predicted Class'])
plt.figure(figsize = (10,5))
plt.title('Confusion Matrix')
sb.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
print('Test Accuracy:',metrics.accuracy_score(df_test['Retinopathy grade'], df_test['Predicted Class']))
