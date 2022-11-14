__authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "03/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# IMPORTS
import os
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_io as tfio
import pickle as pkl
import pandas as pd
import numpy as np
import src.cnn as cnn
import src.inception as inception
from sklearn.utils import class_weight


with open('data/esm2_8M_dict.p', "rb") as pickle_file:
    database_dict = pkl.load(pickle_file)  # open data

data = pd.DataFrame.from_dict({(i): database_dict[i]
                               for i in database_dict.keys()},
                              orient='index')  # Transform nested dict into dataframe


x = data["x"]  # Extract features from the dataframe

y = data["y"]  # Extract Vector to predict from the dataframe

sample_weights = data["sample_weight"]  # Extract sample weight

length = max(map(len, x))  # find the max len seq = 5126


# Create a zero matrice for inputs
matri = np.zeros((8196, 5126, 320), dtype=np.float16)

# Create a zero matrice for outputs
to_pred = np.zeros((8196, 5126), dtype=int)


for i in range(x.shape[0]):
    seq_len = x[i].shape[0]
    matri[i][0:seq_len] = x[i]
    to_pred[i][0:seq_len] = y[i]


matri = matri[:, 0:512, :]
# print(f"la taille est de {matri.shape}")
to_pred = to_pred[:, 0:512]
# print(f"la taille est de {to_pred.shape}")


X = np.asarray(matri)

h5f_x = h5py.File('data/X.h5', 'w')
h5f_x.create_dataset('X', data=X)
h5f_x.close()


h5f_y = h5py.File('data/to_pred.h5', 'w')
h5f_y.create_dataset('to_pred', data=to_pred)
h5f_y.close()


# Transform hdf5 into tf.data.Dataset
X_learn = tfio.IODataset.from_hdf5(X, dataset="data/X.h5")
Y_learn = tfio.IODataset.from_hdf5(to_pred, dataset="data/to_pred")

learn = tf.data.Dataset.zip((X_learn, Y_learn)).batch(
    100).prefetch(tf.data.experimental.AUTOTUNE)  # , sample_weights


model = cnn.inception()


history = model.fit(learn, validation_split=0.2, epochs=2, batch_size=5)
