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
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping


# Save model name each k fold

def get_model_name(k):
    return 'model_'+str(k)+'.h5'
################################


# SELECTION DU MODEL: Choisir 1 seul Modèle.
model_cnn = cnn.cnn()

# # model_inc = inception.inception()

# # model_gru = gru.gru()


###### Chargement du jeu Test ################
X_test = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/x_test")

# print(X_test)
Y_test = tfio.IODataset.from_hdf5(
    'data/small_database_window13_withfolds.h5', dataset="/y_test")

# print(Y_test)

#################


# VALIDATION_ACCURACY = []
# VALIDATION_LOSS = []

#### Sauvegarde de chaque modèle #################################
save_dir = 'saved_models/'
fold_var = 0

###### Boucle permettant  de créer chaque data set validation et training #################################
for i in range(10):
    list_noi = [*range(10)]
    list_noi.remove(i)
    for j in list_noi:
        X_train = tfio.IODataset.from_hdf5(
            'data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[0]}").concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[1]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[2]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[3]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[4]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[5]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[6]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[7]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/x_train_{list_noi[8]}"))

    Y_train = tfio.IODataset.from_hdf5(
        'data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[0]}").concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[1]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[2]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[3]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[4]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[5]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[6]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[7]}")).concatenate(
            tfio.IODataset.from_hdf5('data/small_database_window13_withfolds.h5', dataset=f"/y_train_{list_noi[8]}"))

    X_val = tfio.IODataset.from_hdf5(
        'data/small_database_window13_withfolds.h5', dataset=f"/x_train_{i}")
    Y_val = tfio.IODataset.from_hdf5(
        'data/small_database_window13_withfolds.h5', dataset=f"/y_train_{i}")


###### Ajout des poids #####
    sample_weights = tfio.IODataset.from_hdf5(
        'data/small_database_window13_withfolds.h5', dataset="/sample_weight")

# Creation des dataset contenant les X , Y et poids  de chaque groupe
    learn = tf.data.Dataset.zip((X_train, Y_train, sample_weights)).shuffle(1000).batch(
        100).prefetch(tf.data.experimental.AUTOTUNE)

    train = tf.data.Dataset.zip((X_test, Y_test, sample_weights)).batch(
        100).prefetch(tf.data.experimental.AUTOTUNE)

    validation = tf.data.Dataset.zip((X_val, Y_val, sample_weights)).batch(
        100).prefetch(tf.data.experimental.AUTOTUNE)

# CREATION DE CALLBACKS
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var),
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')  # Ce callback permet de conserver le meilleur modele a chaque iteration.

    # Ce callback permet de stopper les epoch lorsque les performance n'avance plus.
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=0, mode='min')

    callbacks_list = [checkpoint, earlyStopping]

### Dict des parametres a ester pendant la CV
    param_grid = {
        'optimizer': ['rmsprop', 'adam', 'sgd'],
        'epochs': [100, 150, 200]}


####### Recherche des parametres optimaux

    model = KerasClassifier(build_fn=model_cnn, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(learn, validation_data=validation,
                           callbacks=callbacks_list, shuffle=True)  # application des param optimaux
#################

    print('---- GRID SEARCH RESULTS ----')
    for p, s in zip(grid_result.cv_results_['params'], grid_result.cv_results_['mean_test_score']):
        print(
            f' Accuracy : {round(s*100,2)} % | Param : {p} | model : {fold_var}')

        # PLOT HISTORY
    # :
    # :

    # LOAD BEST MODEL to evaluate the performance of the model

    # model.load_weights("saved_models/model_"+str(fold_var)+".h5")
    # results = model.evaluate(train)
    # results = dict(zip(model.metrics_names,results))

    # VALIDATION_ACCURACY.append(results['accuracy'])
    # VALIDATION_LOSS.append(results['loss'])

    tf.keras.backend.clear_session()

    fold_var += 1
