__authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
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
# import src.gru as gru
from sklearn.utils import class_weight
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold



# # ######### Choisir 1 seul Modèle.
model = cnn.cnn()

# # model = inception.inception()

# # model = gru.gru()

# #################

###### Chargement du jeu Test ################
X_test = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/x_test")

print(X_test)
Y_test = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/y_test")

print(Y_test)

#################



for i in range(10):
    X_train = tfio.IODataset.from_hdf5(
        'data/small_database_window13_withfolds.h5', dataset="/x_train")

    Y_train = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/y_train")

    sample_weights = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/sample_weight")
    
    X_val = tfio.IODataset.from_hdf5(
        'data/small_database_window13_withfolds.h5', dataset=f"/x_train_{i}")
    Y_val = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset=f"/y_train_{i}")
    
    learn = tf.data.Dataset.zip((X_train, Y_train, sample_weights)).batch(
        100).prefetch(tf.data.experimental.AUTOTUNE)

    train = tf.data.Dataset.zip((X_test, Y_test, sample_weights)).batch(
        100).prefetch(tf.data.experimental.AUTOTUNE)
    
    validation = tf.data.Dataset.zip((X_val, Y_val, sample_weights)).batch(
        100).prefetch(tf.data.experimental.AUTOTUNE)


    history = model.fit(learn, epochs=20, validation_data=validation)

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Test'], loc='upper left')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# model.evaluate(train)
