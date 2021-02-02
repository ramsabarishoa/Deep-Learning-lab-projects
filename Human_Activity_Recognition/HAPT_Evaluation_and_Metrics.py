acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(24, 8))
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
results = mdl.evaluate(test_ds)

print('test loss, test acc:', results)

#Confusion Matrix 
y_pred = []
y_true = []

#Get predicted labels for entire test dataset
for files, labels in test_ds.take(-1):
    result_predict = mdl.predict(files)
    result_predict = np.asarray(result_predict)
    result_predict = np.argmax(result_predict, axis=2)
    y_pred.append(np.array(result_predict).reshape(result_predict.size, 1))
    label_t = np.argmax(tf.convert_to_tensor(labels), axis=2)
    y_true.append(np.array(label_t).reshape(label_t.size, 1)) #true label
    
y_pred = np.asarray(y_pred)
y_true = np.asarray(y_true)

y_pred_concat = []
y_true_concat = []

#Create numpy array of predicted and true labels
for index in range(0, len(y_pred)):

  if (index == 0):
    y_pred_concat = y_pred[index]
    y_true_concat = y_true[index]

  else:
    y_pred_concat = np.concatenate((y_pred_concat, y_pred[index]), axis=0)
    y_true_concat = np.concatenate((y_true_concat, y_true[index]), axis=0)

print(len(y_pred_concat))
print(len(y_true_concat))

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import math
from sklearn.metrics import classification_report, confusion_matrix

labels=['WALKING',
'WALKING_UPSTAIRS',
'WALKING_DOWNSTAIRS',
'SITTING',
'STANDING',
'LAYING',
'STAND_TO_SIT',
'SIT_TO_STAND',
'SIT_TO_LIE',
'LIE_TO_SIT',
'STAND_TO_LIE',
'LIE_TO_STAND'
]

#Confusion matrix 
confusion_matrix = metrics.confusion_matrix(y_true_concat, y_pred_concat)
#Normalized confusion matrix 
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix) * 100

#Plot confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('Ground Truth')
plt.xlabel('Predicted label')
title = 'Confusion matrix for HAPT dataset'
plt.show();

#Plot normalized confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(normalised_confusion_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt='0.2g' );
plt.title("Normalized Confusion matrix")
plt.ylabel('Ground Truth')
plt.xlabel('Predicted label')
title = 'HAR Normalized Confusion matrix'
plt.show();

print(classification_report(y_true_concat, y_pred_concat, target_names=labels))