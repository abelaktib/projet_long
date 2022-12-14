import src.inception as inception
from keras.callbacks import EarlyStopping,  ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import  cross_val_score
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import src.cnn as cnn
import src.simple as simple
import numpy as np
import pandas as pd
import tensorflow_io as tfio
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import random
from sklearn import metrics

__authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "16/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# # IMPORTS
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
SEED = 4
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# SELECTION DU MODEL: Choisir 1 seul Modèle.


# model_inc = inception.inception()

# model_gru = gru.gru()


##### Chargement des dataset ################

f = h5py.File(args.file, 'r')
print(list(f.keys()))

ytrain = f['Y_training'][:]

X_train = tfio.IODataset.from_hdf5(args.file, dataset=f"/X_training")
Y_train = tfio.IODataset.from_hdf5(args.file, dataset=f"/Y_training")

# #
X_val = tfio.IODataset.from_hdf5(args.file, dataset=f"/X_validation")
Y_val = tfio.IODataset.from_hdf5(args.file, dataset=f"/Y_validation")
# #

# ###### Ajout des poids #####
sw_training = tfio.IODataset.from_hdf5(
    args.file, dataset="/sample_weights_training")  # ample_weights
sw_validation = tfio.IODataset.from_hdf5(
    args.file, dataset="/sample_weights_validation")

# # Creation des dataset contenant les X , Y et poids  de chaque groupe
learn = tf.data.Dataset.zip((X_train, Y_train, sw_training)).batch(
    64).prefetch(tf.data.experimental.AUTOTUNE)


validation = tf.data.Dataset.zip((X_val, Y_val, sw_validation)).batch(
    64).prefetch(tf.data.experimental.AUTOTUNE)


x_val = tf.data.Dataset.zip((X_val)).batch(64).prefetch(
    tf.data.experimental.AUTOTUNE)  # For callbacks y_pred
x_learn = tf.data.Dataset.zip((X_train)).batch(64).prefetch(
    tf.data.experimental.AUTOTUNE)
y_learn = tf.data.Dataset.zip((Y_train)).batch(64).prefetch(
    tf.data .experimental.AUTOTUNE)
y_val = tf.data.Dataset.zip((Y_val)).batch(64).prefetch(
    tf.data .experimental.AUTOTUNE)

y_target_iter = np.concatenate([i for i in y_learn.as_numpy_iterator()])


yval_target_iter = np.concatenate([i for i in y_val.as_numpy_iterator()])
# y_target_batch = next(y_target_iter)
# sw_training


###### CALLBACKS#######
# CREATION DE CALLBACKS
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var),
                                                monitor='val_accuracy', verbose=1,
                                                save_best_only=True, mode='max')  # Ce callback permet de conserver le meilleur modele a chaque iteration.


# Dallback qui va lancer des predictions et créer des matrices de confusion apres l'apprentissage terminé
class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, y_target_iter, yval_target_iter, lr):
        super()
        self.y_target_iter = y_target_iter
        self.yval_target_iter = yval_target_iter
        self.compteur = 0
        self.lr = lr

    def on_train_end(self, logs=None):
        y_pred = self.model.predict(x_learn)
        # print(y_pred)
        uninq_ypred = tf.unique_with_counts(tf.math.argmax(y_pred, axis=1))
        y_vali = self.model.predict(x_val)
        # uninq_yvali = tf.unique_with_counts(tf.math.argmax(y_vali, axis=1))

        ############ MATRIX DE CONFUSION  #############
        tn, fp, fn, tp = metrics.confusion_matrix(
            y_target_iter.argmax(1), y_pred.argmax(1)).ravel()
        # print('prediction: {} at epoch: {} {}'.format(y_pred, epoch, uninq_ypred))
        # print([_.shape for _ in self.y_target_iter])
        disp = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(
            y_target_iter.argmax(1), y_pred.argmax(1)))
        disp.plot()
        plt.savefig('figure/learn/confusion_matrix' +
                    str(self.compteur)+"_" + str(lr)+".png")
        disp2 = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(
            y_target_iter.argmax(1), y_pred.argmax(1), normalize='true'))
        disp2.plot(values_format='.3f')
        plt.savefig('figure/learn/confusion_matrix_normalize'
                    + str(self.compteur)+"_" + str(self.lr)+".png")

        print("####################################################")
        with open("confusion_matrix_" + str(self.lr) + ".csv", "a", encoding="utf-8") as file_m:
            file_m.write(
                "TRUE_NEGATIF, FALSE_POSITIF, FALSE_NEGATIF, TRUE_POSITIF\n")
            file_m.write(f"{tn},{fp},{fn},{tp}\n")

        print("####################################################")

        ############ MATRIX DE CONFUSION VAL #############
        tnv, fpv, fnv, tpv = metrics.confusion_matrix(
            yval_target_iter.argmax(1), y_vali.argmax(1)).ravel()
        # print('prediction: {} at epoch: {} {}'.format(y_vali, epoch, uninq_ypred))
        # print([_.shape for _ in self.y_target_iter])
        disp3 = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(
            yval_target_iter.argmax(1), y_vali.argmax(1)))
        disp3.plot()
        plt.savefig('figure/val/confusion_val_matrix' +
                    str(self.compteur)+"_" + str(self.lr)+".png")

        disp4 = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(
            yval_target_iter.argmax(1), y_vali.argmax(1), normalize='true'))
        disp4.plot(values_format='.3f')
        plt.savefig('figure/val/confusion_matrix_val_normalize'
                    + str(self.compteur)+"_" + str(self.lr)+".png")

        print("####################################################")
        with open("confusion_matrix_val" + str(self.lr) + ".csv", "a", encoding="utf-8") as file_t:
            file_t.write(
                "TRUE_NEGATIF_VAL, FALSE_POSITIF_VAL, FALSE_NEGATIF_VAL, TRUE_POSITIF_VAL\n")
            file_t.write(f"{tnv},{fpv},{fnv},{tpv}\n")
        print("####################################################")

        self.compteur += 1



############
############ COURBE ROC #############œ
        # ROC curve
        fpr, tpr, tresholds = metrics.roc_curve(y_target_iter.argmax(1), y_pred.argmax(1))
        roc_auc = metrics.auc(fpr, tpr)
        roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        roc_display.plot()
        roc_display.figure_.savefig('figure/roc'+
                    str(self.compteur)+"_" + str(self.lr)+".jpeg")

        # PR Curve
        precision, recall, _ = metrics.precision_recall_curve(y_target_iter.argmax(1), y_pred.argmax(1))
        pr_display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        pr_display.plot()
        pr_display.figure_.savefig('figure/pr'+
                    str(self.compteur)+"_" + str(self.lr)+".jpeg")

# Callback qui va reduire le lr
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=80, min_lr=0)

# Callback qui va arreter l apprentissage
earlyStopping = EarlyStopping(
    monitor='val_loss', patience=20, verbose=0, mode='min')

# Callback qui creer le csv de performance
csv_logger = CSVLogger('log.csv', append=True, separator=';')


########## Class WEIGHTING #################################
print("#############################################################")


def calculating_class_weights(y_true):
    weights = compute_class_weight(
        'balanced',
        classes=[0., 1.],
        y=y_true[:].argmax(axis=1))
    return weights


class_weights = calculating_class_weights(ytrain)
print(class_weights)

class_weights = {0: class_weights[0], 1: class_weights[1]}

print("#############################################################")
print("#############################################################")
print("#############################################################")


# # list des batchsize a tester
batch_size = 64
EPOCHS = 20
learning_rate_list = [1e-4]  # , 1e-10, 1e-12,1e-15,1e-20

for lr in learning_rate_list:
    callbacks_list = [reduce_lr, PredictionCallback(
        y_target_iter, yval_target_iter, lr), checkpoint, csv_logger]  # , checkpoint,earlyStopping

    model_cnn = simple.simple(lr)

    history = model_cnn.fit(
        learn, validation_data=validation,
        epochs=EPOCHS,
        batch_size=64,
        callbacks=callbacks_list, class_weight=class_weights,
        shuffle=True
    )
# fold_var += 1
# print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.legend(['Train'], loc='upper left')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()
# plt.plot(history.history['loss'])
# plt.show()
