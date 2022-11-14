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
import src.inception as inception
import src.gru as gru
from sklearn.utils import class_weight
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold

# kfold = KFold(n_splits=num_folds, shuffle=True)
# for train, test in kfold.split(inputs, targets):

X_train = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/x_train")
print(X_train)
Y_train = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/y_train")
print(Y_train)
sample_weights = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/sample_weight")
print(sample_weights)
####### Make group for K folds
hf = h5py.File('data/small_database_window13_withfolds.h5', 'r')
groups = np.array(hf['/folds'][:]) 

print(groups)

# logo = LeaveOneGroupOut()

group_kfold = GroupKFold(n_splits=10)

group_kfold.get_n_splits(X_train,Y_train, groups)

group_kfold.get_n_splits(groups=groups)

for train_index, val_index in group_kfold.split(X_train,Y_train, groups):
    print("TRAIN:", train_index, "TEST:", val_index)













# X_test = tfio.IODataset.from_hdf5(
#     'data/small_database_window13_withfolds.h5', dataset="/x_test")
# print(X_test)
# Y_test = tfio.IODataset.from_hdf5(
#     'data/small_database_window13_withfolds.h5', dataset="/y_test")
# print(Y_test)


# learn = tf.data.Dataset.zip((X_train, Y_train, sample_weights)).batch(
#     100).prefetch(tf.data.experimental.AUTOTUNE)

# train = tf.data.Dataset.zip((X_test, Y_test, sample_weights)).batch(
#     100).prefetch(tf.data.experimental.AUTOTUNE)

# # ######### Choisir 1 seul Modèle.
# model = cnn.cnn()

# # model = inception.inception()

# # model = gru.gru()

# #################

# history = model.fit(learn, epochs=20)

# print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.legend(['Train'], loc='upper left')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()
# plt.plot(history.history['loss'])
# plt.show()

# model.evaluate(train)
