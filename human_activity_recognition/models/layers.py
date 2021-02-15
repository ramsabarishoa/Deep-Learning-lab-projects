'''Model Architecture'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from input_pipeline.data_preprocessing_visualization import N_WindowSize, N_Features, N_CLASSES

print('------------------')
print('Model Architecture')
print('------------------')

def model(input_shape, n_classes):
    """Defining the model architecture.
       Parameters:
           input_shape (tuple: 2): input shape of the neural network
           n_classes   (int): number of classes, corresponding to the number of output neurons
       Returns:
           (keras.Model): keras model object
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(256, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.10)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    mdl = keras.Model(inputs=inputs, outputs=outputs, name='HAR_Model')

    mdl.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer='RMSProp',
                metrics=['accuracy'])
    return mdl

#Passing the WindowSize and Features
#Classes is 12 here
mdl = model((N_WindowSize, N_Features), N_CLASSES)

print('------------------------------------------Start------------------------------------------')
print('Printing the Model Summary')
print(mdl.summary())
print('------------------------------------------End------------------------------------------')
