import tensorflow as tf
from input_pipeline.preprocessing import train_generator, val_generator
from models.layers import mdl

def Trainer(epochs):
    """Function to train the compiled model based on the dataset inputs and number of epochs"""
    epochs = epochs
    history = mdl.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    mdl.save('/home/swetha/PycharmProjects/DL_Code_Integration/dl-lab-2020-team05/experiments/DRD_Model.h5')

    return history
print(''' ***************************Start Training************************''')
epochs = 200

history = Trainer(epochs)
print(''' ***************************End Training************************''')
