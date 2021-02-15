import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import shutil

print('Pandas Version:', pd.__version__)
print('Numpy Version:', np.__version__)
print('Tensorflow version:', tf.__version__)

#Defining Constants
SAMPLING_FREQUENCY = 50
NOISY_ROWS = 250
N_Features = 6
N_WindowSize = 250
N_WindowShift = 125
N_prefetch = 8
BATCH_SIZE = 16
N_ShuffleBuffer = 200
N_CLASSES = 12

'''The activities are listed in activity_labels.txt
The activities are three static activities (sitting, standing, lying), 
three dynamic activities (walking, walking downstairs, walking upstairs), 
and six postural activities (stand-to-sit, sit-to-stand, lie-to-stand,stand-to-lie, stand-to-sit, sit-to-stand).'''

data_set_path = input('Enter the path for the dataset. Please unzip the contents of the dataset before loading the folder path: ')
Debug = '''The Debug option here helps in visualizing and analysing the data(train and test).'''
print(Debug)
debug_mode_input_pipeline = int(input('Enter 1 for enabling debug option(DEBUG_INPUT_PIPELINE) else enter 0 : '))
Activity_Labels = pd.read_csv( data_set_path + '/activity_labels.txt',
                              delimiter= '\s+', index_col=False, names=['label', 'activity'])

#The training IDs of the subjects is printed here with the total number of training samples (21 Subjects)
train_ids = pd.read_csv(data_set_path + '/Train/subject_id_train.txt',
                      sep=" ", header=None, names=['sub_train'])

#The testing IDs of the subjects is printed here with the total number of testing samples (9 Subjects)
test_ids = pd.read_csv(data_set_path + '/Test/subject_id_test.txt',
                      sep=" ", header=None, names=['sub_test'])

#Debug
if debug_mode_input_pipeline == 1:
  print(Activity_Labels)
  print('The subject IDs used for training:',np.unique(train_ids['sub_train'].to_list()))
  print('Total number of subjects used for training:', len(np.unique(train_ids['sub_train'].to_list())))
  print('---------------------------------------------------------------------------------------------')
  print('The subject IDs used for training:', np.unique(test_ids['sub_test'].to_list()))
  print('Total number of subjects used for training:', len(np.unique(test_ids['sub_test'].to_list())))
  print('---------------------------------------------------------------------------------------------')

# The labels.txt contains the start and end frame for the 12 activities along with the userID and experiment number
df_label = pd.read_csv(data_set_path + '/RawData/labels.txt', sep=" ",
                       header=None, names=['Exp_number_ID', 'User_number_ID', 'Activity_number_ID', 'LStartp', 'LEndp'])

def data_visualization():
  Vis_Text = '''==========================================================================================
  Visualization of the signals can be found here with enabling the debug_mode_input_pipeline==1
  Visualization of the training and test data set
  ============================================================================================='''
  print(Vis_Text)
  # Accelerometer Train data for user01_exp01 for all 12 activities
  acc_train = pd.read_csv(data_set_path + '/RawData/acc_exp01_user01.txt',
                          delimiter='\s+', names=['x-axis', 'y-axis', 'z-axis'])

  # Gyroscope Train data for user01_exp01 for all 12 activities
  gyc_train = pd.read_csv(data_set_path + '/RawData/gyro_exp01_user01.txt',
                          delimiter='\s+', names=['x-axis', 'y-axis', 'z-axis'])

  # Accelerometer Train data for user20_exp01 for all 12 activities
  acc_test = pd.read_csv(data_set_path + '/RawData/acc_exp37_user18.txt',
                         delimiter='\s+', names=['x-axis', 'y-axis', 'z-axis'])

  # Gyroscope Train data for user20_exp01 for all 12 activities
  gyc_test = pd.read_csv(data_set_path + '/RawData/gyro_exp37_user18.txt',
                         delimiter='\s+', names=['x-axis', 'y-axis', 'z-axis'])

  if debug_mode_input_pipeline == 1:
    print(acc_train)
    print(gyc_train)
    print(acc_test)
    print(gyc_test)

    ax_acc_train = acc_train[["x-axis", "y-axis", "z-axis"]].plot(figsize=(20, 4))
    ax_acc_train.set_title('Accelerometer data for User01_Exp01')
    plt.show()
    ax_gyc_train = gyc_train[["x-axis", "y-axis", "z-axis"]].plot(figsize=(20, 4))
    ax_gyc_train.set_title('Gyroscope data for User01_Exp01')
    plt.show()

    ax_acc_test = acc_test[["x-axis", "y-axis", "z-axis"]].plot(figsize=(20, 4))
    ax_acc_test.set_title('Accelerometer data for User18_Exp37')
    plt.show()
    ax_gyc_test = gyc_test[["x-axis", "y-axis", "z-axis"]].plot(figsize=(20, 4))
    ax_gyc_test.set_title('Gyroscope data for User18_Exp37')
    plt.show()

  Vis_Text_1 = '''==========================================================================================
  Visualization of the 12 activities can be found here with enabling the debug_mode_input_pipeline==1
  This is for the train data
  ============================================================================================='''

  print('\n')
  print(Vis_Text_1)
  print('\n')

  if debug_mode_input_pipeline == 1:
    test_user01 = df_label[df_label['User_number_ID'] == 1]
    test_user01 = test_user01.sort_values(by=['Activity_number_ID'])
    test_user01 = test_user01.drop_duplicates(subset=['Activity_number_ID'], keep='first')
    print(df_label.head())
    # fig, axes = plt.subplots(nrows=4, ncols=3)
    for i in range(len(test_user01)):
      start_point = test_user01['LStartp'].iloc[i]
      end_point = test_user01['LEndp'].iloc[i]
      acc_train[["x-axis", "y-axis", "z-axis"]].iloc[start_point:end_point].plot(figsize=(10, 4),
                                                                                 legend=True,
                                                                                 title=Activity_Labels['activity'].iloc[
                                                                                   i])
      gyc_train[["x-axis", "y-axis", "z-axis"]].iloc[start_point:end_point].plot(figsize=(10, 4),
                                                                                 legend=True,
                                                                                 title=Activity_Labels['activity'].iloc[
                                                                                   i])
def acc_gyro_separation():
  '''Separating the accelerometer and gyroscope data'''

  raw_data_path = glob.glob(data_set_path + "/RawData/*.txt") #All the raw data is available here
  train_rawdata_acc = []
  train_rawdata_gyro = []
  for i in raw_data_path:
    if os.path.basename(i).find('acc_'):
      train_rawdata_gyro.append(i)
    elif os.path.basename(i).find('gyro_'):
      train_rawdata_acc.append(i)

  train_rawdata_gyro = train_rawdata_gyro[:-1]
  #Debug
  print('The total number of acceleration files available: ', len(train_rawdata_gyro))
  print('The total number of gyroscope files available: ', len(train_rawdata_acc))

#Dictionary containing the 12 labels
LABEL_NAMES = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',  # 3 dynamic activities
    4: 'SITTING', 5: 'STANDING', 6: 'LYING',  # 3 static activities
    7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND', 9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT',
    11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND',  # 6 postural Transitions
}

def read_acc_gyro_data(file_path, column_names):
  df = pd.read_csv(file_path, delimiter=' ',
                   header=None,
                   names=column_names)
  x = column_names[0]
  y = column_names[1]
  z = column_names[2]
  df[x] = df[x].apply(convert_to_float)
  df[y] = df[y].apply(convert_to_float)
  df[z] = df[z].apply(convert_to_float)
  df.dropna(axis=0, how='any', inplace=True)
  return df

#Function to convert all accelerometer and gyroscope values to float
def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan
def data_Normalization():
  df_train = []
  df_test = []
  df_validation = []
  labels_train = []
  labels_test = []
  labels_validation = []
  df_acc = []
  rows_per_exp = []
  raw_data_path_acc_gyro = sorted(glob.glob(data_set_path + '/RawData/*'))
  Raw_acc_path = raw_data_path_acc_gyro[0:61]
  Raw_gyro_path = raw_data_path_acc_gyro[61:122]
  raw_acc_columns = ['acc_X', 'acc_Y', 'acc_Z']
  raw_gyro_columns = ['gyro_X', 'gyro_Y', 'gyro_Z']

  for path_index in range(0, 61):
    user = int(raw_data_path_acc_gyro[path_index][-6:-4]) #Extract user ID
    exp_id = int(raw_data_path_acc_gyro[path_index][-13:-11]) #Extract Experiment ID

    '''Accelerometer***************************************************
    Read, remove noisy rows and normalize the data
    Accelerometer***************************************************'''
    df_acc = read_acc_gyro_data(raw_data_path_acc_gyro[path_index], raw_acc_columns)
    # Remove noisy rows
    df_acc = df_acc.iloc[NOISY_ROWS:]
    df_acc = df_acc.iloc[:-NOISY_ROWS]
    # Normalize the data using Z-Score normalization
    df_acc['acc_X'] = (df_acc['acc_X'] - df_acc['acc_X'].mean()) / \
                      df_acc['acc_X'].std(ddof=0)
    df_acc['acc_Y'] = (df_acc['acc_Y'] - df_acc['acc_Y'].mean()) / \
                      df_acc['acc_Y'].std(ddof=0)
    df_acc['acc_Z'] = (df_acc['acc_Z'] - df_acc['acc_Z'].mean()) / \
                      df_acc['acc_Z'].std(ddof=0)

    df_acc = df_acc.round({'acc_X': 4, 'acc_Y': 4, 'acc_Z': 4}) #Rounding and formatting the data values

    '''Gyroscope***************************************************
      Read, remove noisy rows and normalize the data
      Gyroscope***************************************************'''
    df_gyro = read_acc_gyro_data(raw_data_path_acc_gyro[path_index + 61], raw_gyro_columns)
    df_gyro = df_gyro.iloc[NOISY_ROWS:]
    df_gyro = df_gyro.iloc[:-NOISY_ROWS]
    # Normalize data using Z-Score normalization
    df_gyro['gyro_X'] = (df_gyro['gyro_X'] - df_gyro['gyro_X'].mean()) / \
                        df_gyro['gyro_X'].std(ddof=0)
    df_gyro['gyro_Y'] = (df_gyro['gyro_Y'] - df_gyro['gyro_Y'].mean()) / \
                        df_gyro['gyro_Y'].std(ddof=0)
    df_gyro['gyro_Z'] = (df_gyro['gyro_Z'] - df_gyro['gyro_Z'].mean()) / \
                        df_gyro['gyro_Z'].std(ddof=0)
    df_gyro = df_gyro.round({'gyro_X': 4, 'gyro_Y': 4, 'gyro_Z': 4})  #Rounding and formatting the data values

    # Concatenation of accelerometer and gyro data
    df_signals = pd.concat([df_acc, df_gyro], axis=1)
    df_signals_numpy = df_signals.to_numpy()
    # The unlabelled data is set to 0
    labels = np.zeros(len(df_signals_numpy))

    for index, rows in df_label.iterrows():
      if rows['Exp_number_ID'] == exp_id:
        start = rows['LStartp']
        end = rows['LEndp']
        label_value = int(rows['Activity_number_ID'])
        labels[start - NOISY_ROWS:end - NOISY_ROWS] = label_value

    #Training data
    if 1 <= user <= 21:
      labels_train.append(labels)
      for row in df_signals_numpy:
        row = row.reshape(1, 6).flatten()
        df_train.append(row)

    #Test data
    elif 22 <= user <= 27:
      labels_test.append(labels)
      for row in df_signals_numpy:
        row = row.reshape(1, 6).flatten()
        df_test.append(row)

    #Validation data
    elif 28 <= user <= 30:
      labels_validation.append(labels)
      for row in df_signals_numpy:
        row = row.reshape(1, 6).flatten()
        df_validation.append(row)

  labels_train = np.concatenate(labels_train).astype('int32')
  labels_test = np.concatenate(labels_test).astype('int32')
  labels_validation = np.concatenate(labels_validation).astype('int32')

  df_train_subset = df_train[0:20000]
  len_df = len(df_train_subset)
  #Debug
  #print(len_df)
  print(len(labels_train), len(labels_test), len(labels_validation))

  #One hot encoding of the labels
  def one_hot_encoding(n_labels):
    one_hot_labels = []
    x = np.zeros(N_CLASSES)
    #print(np.size(n_labels))
    for i in range(0, np.size(n_labels)):
      x = np.zeros(N_CLASSES)
      if n_labels[i] == 1:
        x[0] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 2:
        x[1] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 3:
        x[2] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 4:
        x[3] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 5:
        x[4] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 6:
        x[5] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 7:
        x[6] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 8:
        x[7] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 9:
        x[8] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 10:
        x[9] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 11:
        x[10] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 12:
        x[11] = 1
        one_hot_labels.append(x)
      elif n_labels[i] == 0:
        one_hot_labels.append(x) #Unlabeled Data is marked as zero
      else:
        print('Unrecognized label')

    return one_hot_labels

  train_labels_one_hot = one_hot_encoding(labels_train)
  test_labels_one_hot = one_hot_encoding(labels_test)
  validation_labels_one_hot = one_hot_encoding(labels_validation)
  #Debug
  print(len(train_labels_one_hot), len(test_labels_one_hot), len(validation_labels_one_hot))

  #Sliding window technique is used
  #Shifts by 125 samples (with 50% overlap).
  def sliding_window(files, labels, window_size, window_overlap):
    lfiles = []
    llabels = []
    for i in range(0, int(len(files) / window_overlap) - 1):
      lfiles.append(files[i * window_overlap:i * window_overlap + window_size])
      llabels.append(labels[i * window_overlap:i * window_overlap + window_size])
    return lfiles, llabels

  train_files, train_labels = sliding_window(df_train, train_labels_one_hot, N_WindowSize, N_WindowShift)
  test_files, test_labels = sliding_window(df_test, test_labels_one_hot, N_WindowSize, N_WindowShift)
  validation_files, validation_labels = sliding_window(df_validation, validation_labels_one_hot, N_WindowSize, N_WindowShift)

  def build_dataset(files, labels):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.shuffle(N_ShuffleBuffer)
    ds = ds.prefetch(N_prefetch)
    return ds

  #Build the dataset
  train_ds = build_dataset(train_files, train_labels)
  test_ds = build_dataset(test_files, test_labels)
  validation_ds = build_dataset(validation_files, validation_labels)

  #Debug
  print(len(train_ds))
  print(len(train_files))
  return train_ds, test_ds, validation_ds
data_visualization() #Function to visualize the data
acc_gyro_separation() #Function to separate the accelerometer and gyroscope data

#Function to remove noisy rows and normalize the data
#Returns the train, test and validation dataset

r_train_ds, r_test_ds, r_val_s = data_Normalization()
