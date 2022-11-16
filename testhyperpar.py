__authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "16/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# IMPORTS
import random
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd
import numpy as np
import src.cnn as cnn
# import src.inception as inception
# import src.gru as gru
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import class_weight
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.metrics import get_scorer_names

#
random.seed(10)
#
#
# SELECTION DU MODEL: Choisir 1 seul Modèle.
model_cnn = cnn.cnn()
#
# # model_inc = inception.inception()
#
# # model_gru = gru.gru()
#
#
###### Chargement ddes dataset ################

list_noi = list(range(10))
list_noi.remove(1)

# FILE = 'data/small_database_window13_withfolds.h5'
FILE = '/home/madeleine/gheeraert/SHARED/sugarpred_window13_10folds.h5'
X_train = tfio.IODataset.from_hdf5(FILE, dataset=f"/x_train_{list_noi[0]}")

Y_train = tfio.IODataset.from_hdf5(FILE, dataset=f"/y_train_{list_noi[0]}")

for j in range(1, 9, 1):
    X_train.concatenate(tfio.IODataset.from_hdf5(FILE, dataset=f"/x_train_{list_noi[j]}"))
    Y_train.concatenate(tfio.IODataset.from_hdf5(FILE, dataset=f"/y_train_{list_noi[j]}"))

# #
X_val = tfio.IODataset.from_hdf5(FILE, dataset=f"/x_train_{1}")
Y_val = tfio.IODataset.from_hdf5(FILE, dataset=f"/y_train_{1}")
#

###### Ajout des poids #####
sample_weights = tfio.IODataset.from_hdf5(FILE, dataset="/sample_weight")

# Creation des dataset contenant les X , Y et poids  de chaque groupe
learn = tf.data.Dataset.zip((X_train, Y_train, sample_weights)).shuffle(
    50).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
validation = tf.data.Dataset.zip((X_val, Y_val, sample_weights)).shuffle(
    50).batch(1).prefetch(tf.data.experimental.AUTOTUNE)


# CREATION DE CALLBACKS

# Ce callback permet de stopper les epoch lorsque les performance n'avance plus.
earlyStopping = EarlyStopping(
    monitor='val_loss', patience=10, verbose=0, mode='min')
#
callbacks_list = [earlyStopping]
#
# list des batchsize a tester

batch_size = [50, 100, 200, 250, 300]
EPOCHS = 200

with open("history.csv", "w", encoding="utf-8") as file:
    file.write("BATCH,ACCURACY,VAL_ACCURACY,LOSS,VAL_LOSS\n")
    
    for batch in batch_size:
        history = model_cnn.fit(
            learn, validation_data=validation,
            callbacks=callbacks_list,
            epochs=EPOCHS,
            batch_size=batch
        )

        for e in range(EPOCHS):
            file.write(f"{batch},{history.history['accuracy'][e]},"
                   f"{history.history['val_accuracy'][e]},"
                   f"{history.history['loss'][e]},"
                   f"{history.history['val_loss'][e]}\n")
