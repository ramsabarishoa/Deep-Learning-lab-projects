'''Plotting the training and validation accuracy and loss
'''
from models.layers import mdl
from train import EPOCHS, history
import matplotlib.pyplot as plt
from input_pipeline.data_preprocessing_visualization import r_test_ds

def evaluate(mdl):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(20, 7))
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
    return

evaluate(mdl)

#Evaluate the model
results = mdl.evaluate(r_test_ds)
print('test loss, test acc:', results)
