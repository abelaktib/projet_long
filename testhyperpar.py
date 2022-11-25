from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
import argparse
import src.cnn as cnn
import numpy as np
import pandas as pd
import tensorflow_io as tfio
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import random

_authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "16/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# # IMPORTS
# import src.inception as inception
# import src.gru as gru

tf.config.list_physical_devices('GPU')


##### argparse #################################
parse = argparse.ArgumentParser()
parse.add_argument("file")
args = parse.parse_args()
args.file


# Save model name
#
save_dir = 'saved_models/'
fold_var = 0


def get_model_name(k):
    return 'model_bigdata'+str(k)+'.h5'


# Random seed
tf.keras.utils.set_random_seed(42)

# SELECTION DU MODEL: Choisir 1 seul Modèle.
model_cnn = cnn.cnn()

# model_inc = inception.inception()

# model_gru = gru.gru()


##### Chargement des dataset ################

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
learn = tf.data.Dataset.zip((X_train, Y_train, sw_training)).batch(
    64).prefetch(tf.data.experimental.AUTOTUNE)
validation = tf.data.Dataset.zip((X_val, Y_val, sw_training)).batch(
    64).prefetch(tf.data.experimental.AUTOTUNE)
val = tf.data.Dataset.zip((X_val)).batch(64).prefetch(
    tf.data.experimental.AUTOTUNE)  # For callbacks y_pred


###### CALLBACKS#######
# CREATION DE CALLBACKS
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var),
                                                monitor='val_accuracy', verbose=1,
                                                save_best_only=True, mode='max')  # Ce callback permet de conserver le meilleur modele a chaque iteration.


# class PredictionCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         y_pred = self.model.predict(val)
#         print('prediction: {} at epoch: {}'.format(y_pred, epoch))


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0)


callbacks_list = [reduce_lr, checkpoint]  #PredictionCallback(),

class_weights = {0: 0.96, 1: 0.04}

# # list des batchsize a tester
batch_size = [64]
EPOCHS = 20
with open("historybigdata_seed.csv", "w", encoding="utf-8") as file:
    file.write("EPOCHS,BATCH,ACCURACY,VAL_ACCURACY,LOSS,VAL_LOSS,ROC,PR,VAL_ROC,VAL_PR,PRECISION,RECALL,VAL_PRECISION,VAL_RECALL,BIN_ACC,VAL_BIN_ACC,LR,TRUE_POSITIVES,TRUE_NEGATIVES,FALSE_POSITIVES,FALSE_NEGATIVES,VAL_TRUE_POSITIVES,VAL_TRUE_NEGATIVES,VAL_FALSE_POSITIVES,VAL_FALSE_NEGATIVES\n")

    for batch in batch_size:

        history = model_cnn.fit(
            learn, validation_data=validation,
            epochs=EPOCHS,
            batch_size=64,
            callbacks=callbacks_list, class_weight=class_weights,
            shuffle=True
        )

        for e in range(EPOCHS):
            file.write(f"{e},{batch},{history.history['accuracy'][e]},"
                       f"{history.history['val_accuracy'][e]},"
                       f"{history.history['loss'][e]},"
                       f"{history.history['val_loss'][e]},"
                       f"{history.history['auc'][e]},"
                       f"{history.history['auc_1'][e]},"
                       f"{history.history['val_auc'][e]},"
                       f"{history.history['val_auc_1'][e]},"
                       f"{history.history['precision'][e]},"
                       f"{history.history['recall'][e]},"
                       f"{history.history['val_precision'][e]},"
                       f"{history.history['val_recall'][e]},"
                       f"{history.history['binary_accuracy'][e]},"
                       f"{history.history['val_binary_accuracy'][e]},"
                       f"{history.history['lr'][e]},"
                       f"{history.history['true_positives'][e]},"
                       f"{history.history['true_negatives'][e]},"
                       f"{history.history['false_positives'][e]},"
                       f"{history.history['false_negatives'][e]},"
                       f"{history.history['val_true_positives'][e]},"
                       f"{history.history['val_true_negatives'][e]},"
                       f"{history.history['val_false_positives'][e]},"
                       f"{history.history['val_false_negatives'][e]}\n"
                       )

fold_var += 1
print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.legend(['Train'], loc='upper left')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()
# plt.plot(history.history['loss'])
# plt.show()
