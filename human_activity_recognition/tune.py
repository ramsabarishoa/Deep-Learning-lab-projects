'''Start Hyper-parameter tuning
- HP_OPTIMIZER
- HP_EPOCHS
- HP_LSTM_NEURONS
- HP_DROPOUT

Visualize the results on tensorboard'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp
from input_pipeline.data_preprocessing_visualization import N_WindowSize, N_Features, N_CLASSES
from input_pipeline.data_preprocessing_visualization import r_train_ds, r_val_s, r_test_ds

#Define the hyperparameters
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'RMSProp']))
HP_LSTM_NEURONS = hp.HParam('LSTMNeurons', hp.Discrete([128,256]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_EPOCHS = hp.HParam('epochs',hp.Discrete([5, 10]))

METRIC_ACCURACY = 'accuracy'

path_hparams = input('Enter the path to save the tuning logs: ')

with tf.summary.create_file_writer(path_hparams + 'logs/hparam_tuning').as_default():
  hp.hparams_config(
      hparams = [HP_LSTM_NEURONS, HP_DROPOUT, HP_OPTIMIZER, HP_EPOCHS],
      metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def HP_mdl():
  inputs = keras.Input(shape=(N_WindowSize,N_Features))
  x = layers.LSTM(hparams[HP_LSTM_NEURONS],return_sequences=True)(inputs)
  x = layers.Dropout(hparams[HP_DROPOUT])(x)
  x = layers.LSTM(128,return_sequences=True)(x)
  outputs = layers.Dense(N_CLASSES, activation='softmax')(x)
  mdl = keras.Model(inputs=inputs, outputs=outputs, name='HAR_model')
  return mdl

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

def train_test_model(hparams):
  mdl = HP_mdl()
  opt = hparams[HP_OPTIMIZER]
  mdl.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])
  #Train model
  history = mdl.fit(r_train_ds,validation_data = r_val_s, epochs=hparams[HP_EPOCHS])
  _, accuracy = mdl.evaluate(r_test_ds)
  return accuracy

session_num = 0
for optimizer in HP_OPTIMIZER.domain.values:
  for lstm_neurons in HP_LSTM_NEURONS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
     for epochs in HP_EPOCHS.domain.values:
        hparams = {
          HP_OPTIMIZER: optimizer,
          HP_LSTM_NEURONS: lstm_neurons,
          HP_DROPOUT:dropout_rate,
          HP_EPOCHS:epochs
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run(path_hparams + 'logs/hparam_tuning/' + run_name, hparams)
        session_num += 1

# Visualizing on tensorboard
# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir /content/drive/MyDrive/logs/hparam_tuning
