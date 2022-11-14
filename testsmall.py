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
# import src.inception as inception
# import src.rnn as rnn
from sklearn.utils import class_weight


X_train =tfio.IODataset.from_hdf5('data/small_database_window13.h5', dataset="/x_train")
# print (X_train)
Y_train = tfio.IODataset.from_hdf5('data/small_database_window13.h5', dataset="/y_train")
# print (Y_train)
sample_weights = tfio.IODataset.from_hdf5('data/small_database_window13.h5', dataset="/sample_weight")
# print (sample_weights)

X_test =tfio.IODataset.from_hdf5('data/small_database_window13.h5', dataset="/x_test")
# print (X_test)
Y_test = tfio.IODataset.from_hdf5('data/small_database_window13.h5', dataset="/y_test")
# print (Y_test)


learn = tf.data.Dataset.zip((X_train, Y_train, sample_weights)).batch(100).prefetch(tf.data.experimental.AUTOTUNE)

train = tf.data.Dataset.zip((X_test, Y_test, sample_weights)).batch(100).prefetch(tf.data.experimental.AUTOTUNE)


model = cnn.cnn()

history = model.fit(learn, epochs=50)

plt.plot(history.history['accuracy'])
plt.legend(['Train'], loc='upper left')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
plt.plot(history.history['loss'])
plt.show()

model.evaluate(train)