__authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "03/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# IMPORTS
import os
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
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

#### Save model name each k fold
def get_model_name(k):
    return 'model_'+str(k)+'.h5'
################################

 ######### Choisir 1 seul Modèle.
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



VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

save_dir = '/saved_models/'
fold_var = 0



for i in range(10):
    list_noi = [*range(10)]
    list_noi.remove(i)
    
    X_train = tfio.IODataset.from_hdf5(
        'data/small_database_window13_withfolds.h5', dataset=f"/x_train") 

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
    
    
    
     ##CREATE CALLBACKS
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
							monitor='val_accuracy', verbose=1, 
							save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    
    history = model.fit(learn, epochs=5, validation_data=validation, callbacks=callbacks_list)
    
    	#PLOT HISTORY
	#		:
	#		:
	
	# LOAD BEST MODEL to evaluate the performance of the model
 
    model.load_weights("/saved_models/model_"+str(fold_var)+".h5")
    results = model.evaluate(train) 
    results = dict(zip(model.metrics_names,results))
	
    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
	
    tf.keras.backend.clear_session()
	
    fold_var += 1
