# -*- coding: utf-8 -*-
"""Human Activity Recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ngydv5qcXMYyjN0DTq-UlGswM6P0vO0C
"""

from google.colab import drive
drive.mount('/content/drive')

#Unzip the dataset to a path in the drive
!unzip -uq "/content/drive/MyDrive/HAPT.zip" -d "/content/drive/MyDrive/Human_Activity"
print('Unzipped the contents to the drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('Pandas Version:', pd.__version__)
print('Numpy Version:', np.__version__)

Activity_Labels = pd.read_csv('/content/drive/MyDrive/Human_Activity/activity_labels.txt', 
                              delimiter= '\s+', index_col=False, names=['label', 'activity'])
print(Activity_Labels)

#Train
train_ids = pd.read_csv('/content/drive/MyDrive/Human_Activity/Train/subject_id_train.txt', 
                      sep=" ", header=None, names=['sub_train'])
print('The subject IDs used for training:',np.unique(train_ids['sub_train'].to_list()))
print('Total number of subjects used for training:', len(np.unique(train_ids['sub_train'].to_list())))

#Test
test_ids = pd.read_csv('/content/drive/MyDrive/Human_Activity/Test/subject_id_test.txt', 
                      sep=" ", header=None, names=['sub_train'])
print('The subject IDs used for training:',np.unique(test_ids['sub_train'].to_list()))
print('Total number of subjects used for training:', len(np.unique(test_ids['sub_train'].to_list())))

#Accelerometer Train for user01
acc = pd.read_csv('/content/drive/MyDrive/Human_Activity/RawData/acc_exp01_user01.txt', 
                  delimiter= '\s+', names=['x-axis', 'y-axis', 'z-axis'])
print(acc)

#Gyroscope Train for user01
gyc = pd.read_csv('/content/drive/MyDrive/Human_Activity/RawData/gyro_exp01_user01.txt', 
                  delimiter= '\s+', names=['x-axis', 'y-axis', 'z-axis'])
print(gyc)

# Plot for user1

ax_acc = acc[["x-axis", "y-axis", "z-axis"]].plot(figsize=(20,4))
ax_acc.set_title('Accelerometer data for User01_Exp01')

ax_gyc = gyc[["x-axis", "y-axis", "z-axis"]].plot(figsize=(20,4))
ax_gyc.set_title('Gyroscope data for User01_Exp01')

#Plotting the 12 different activities for User01
df = pd.read_csv('/content/drive/MyDrive/Human_Activity/RawData/labels.txt', sep=" ", 
                 header=None, names=['Exp_number_ID', 'User_number_ID', 'Activity_number_ID', 'LStartp', 'LEndp'])
#print(df.head(20))

#fig, axes = plt.subplots(nrows=4, ncols=3)
test_user01 = df[df['User_number_ID'] == 1]
test_user01 = test_user01.sort_values(by=['Activity_number_ID'])
test_user01 = test_user01.drop_duplicates(subset=['Activity_number_ID'], keep='first')
#print(test_user01)

for i in range(len(test_user01)):
  start_point = test_user01['LStartp'].iloc[i]
  end_point = test_user01['LEndp'].iloc[i]
  acc[["x-axis", "y-axis", "z-axis"]].iloc[start_point:end_point].plot(figsize=(10,4), 
                                                                       legend=True, title=Activity_Labels['activity'].iloc[i])
