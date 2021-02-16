'''Metrics : Confusion Matrix
Displays the true positives and true negatives'''

from train import mdl
from input_pipeline.data_preprocessing_visualization import r_test_ds, Activity_Labels
import numpy as np
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
print('Importing done')

# Confusion Matrix Plotting
predicted_label = []
ground_truth = []

# Get predicted labels for entire test dataset
for files, labels in r_test_ds.take(-1):
  result_predict = mdl.predict(files)
  result_predict = np.asarray(result_predict)
  result_predict = np.argmax(result_predict, axis=2)
  predicted_label.append(np.array(result_predict).reshape(result_predict.size, 1))
  label_t = np.argmax(tf.convert_to_tensor(labels), axis=2)
  ground_truth.append(np.array(label_t).reshape(label_t.size, 1))  # true label

predicted_label = np.asarray(predicted_label)
ground_truth = np.asarray(ground_truth)

predicted_label_concat = []
ground_truth_concat = []

# Create numpy array of predicted and true labels
for index in range(0, len(predicted_label)):

  if (index == 0):
    predicted_label_concat = predicted_label[index]
    ground_truth_concat = ground_truth[index]

  else:
    predicted_label_concat = np.concatenate((predicted_label_concat, predicted_label[index]), axis=0)
    ground_truth_concat = np.concatenate((ground_truth_concat, ground_truth[index]), axis=0)

print(len(predicted_label_concat))
print(len(ground_truth_concat))

# Confusion matrix
confusion_matrix = metrics.confusion_matrix(ground_truth_concat, predicted_label_concat)
# Normalized confusion matrix
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

# Plot confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, xticklabels=Activity_Labels['activity'], yticklabels=Activity_Labels['activity'], annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('Ground Truth')
plt.xlabel('Predicted label')
plt.title('Confusion matrix')
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(normalised_confusion_matrix, xticklabels=Activity_Labels['activity'], yticklabels=Activity_Labels['activity'], annot=True, fmt='0.2g');
plt.title("Normalized Confusion matrix")
plt.ylabel('Ground Truth')
plt.xlabel('Predicted label')
plt.title('Normalized Confusion matrix')
plt.show()

#Print the classification report
print(classification_report(ground_truth_concat, predicted_label_concat, target_names=Activity_Labels['activity']))
