_authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "16/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# # IMPORTS
import sys
import random
import h5py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd
import numpy as np
import src.cnn as cnn
import argparse
# import src.inception as inception
# import src.gru as gru
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import class_weight
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping,TensorBoard,ReduceLROnPlateau
from sklearn.metrics import get_scorer_names

tf.config.list_physical_devices('GPU') 

parse=argparse.ArgumentParser()
parse.add_argument("file")
args=parse.parse_args()
args.file


random.seed(10)


################ SELECTION DU MODEL: Choisir 1 seul Modèle.
model_cnn = cnn.cnn()

# model_inc = inception.inception()

# model_gru = gru.gru()


##### Chargement ddes dataset ################
# argparse


import h5py
f = h5py.File(args.file, 'r')
print(list(f.keys()))


X_train = tfio.IODataset.from_hdf5(args.file, dataset=f"/X_training")
Y_train = tfio.IODataset.from_hdf5(args.file, dataset=f"/Y_training")

# #
X_val = tfio.IODataset.from_hdf5(args.file, dataset=f"/X_validation")
Y_val = tfio.IODataset.from_hdf5(args.file, dataset=f"/Y_validation")
# #
print(X_train)
print(Y_train)

print(X_val)
print(Y_val)
# ###### Ajout des poids #####
sw_training = tfio.IODataset.from_hdf5(args.file, dataset="/sample_weights_training")
sw_validation = tfio.IODataset.from_hdf5(args.file, dataset="/sample_weights_validation")

# # Creation des dataset contenant les X , Y et poids  de chaque groupe
learn = tf.data.Dataset.zip((X_train, Y_train,sw_training)).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
validation= tf.data.Dataset.zip((X_val,Y_val,sw_training)).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
val = tf.data.Dataset.zip((X_val)).batch(64).prefetch(tf.data.experimental.AUTOTUNE) #For callbacks y_pred


# # #


class PredictionCallback(tf.keras.callbacks.Callback):    
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict(val)
    print('prediction: {} at epoch: {}'.format(y_pred, epoch))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0)

    
    
callbacks_list = [PredictionCallback(),ReduceLROnPlateau()]

class_weights ={0:0.96,1:0.04}

# # list des batchsize a tester
batch_size=[64]
EPOCHS = 20
with open("history2.csv", "w", encoding="utf-8") as file:
    file.write("BATCH,ACCURACY,VAL_ACCURACY,LOSS,VAL_LOSS\n")
    
    for batch in batch_size:
        
        history = model_cnn.fit(
            learn, validation_data=validation,
            epochs=EPOCHS,
            batch_size=64,
              callbacks=callbacks_list,class_weight=class_weights,
          shuffle=True
        )

        for e in range(EPOCHS):
            file.write(f"{batch},{history.history['accuracy'][e]},"
                   f"{history.history['val_accuracy'][e]},"
                   f"{history.history['loss'][e]},"
                   f"{history.history['val_loss'][e]}\n")
