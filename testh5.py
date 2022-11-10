__authors__ = ["BELAKTIB Anas"]
__contact__ = [ "anas.belaktib@etu.u-paris.fr"]
__date__ = "03/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# IMPORTS
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K 
import tensorflow_io as tfio
import pickle as pkl
import pandas as pd
import numpy as np
import src.cnn as cnn
from sklearn.utils import class_weight


# # Ouverture du dataset
# f = h5py.File('data/database_window13.h5')
# X_train = f['x'][...]
# Y_train = f['y'][...]
# sample_weights = f['sample_weight']
# f.close()

X =tfio.IODataset.from_hdf5('data/database_window13.h5', dataset="/x")
Y = tfio.IODataset.from_hdf5('data/database_window13.h5', dataset="/y")

sample_weights = tfio.IODataset.from_hdf5('data/database_window13.h5', dataset="/sample_weight")



learn = tf.data.Dataset.zip((X, Y, sample_weights)).batch(5).prefetch(tf.data.experimental.AUTOTUNE)

model = cnn.cnn()



history = model.fit(learn, epochs=200 )  ###,validation_split=0.02

print(history.history.keys())
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['Train', 'Test'], loc='upper left')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.show()

