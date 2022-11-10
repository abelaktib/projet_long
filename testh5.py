__authors__ = ["BELAKTIB Anas"]
__contact__ = [ "anas.belaktib@etu.u-paris.fr"]
__date__ = "03/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# IMPORTS
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K 
import tensorflow_io as tfio
import pickle as pkl
import pandas as pd
import numpy as np
import src.cnn as cnn
from sklearn.utils import class_weight


# Ouverture du dataset
X = tfio.IODataset.from_hdf5('data/database_window13.h5', dataset="/x")
Y = tfio.IODataset.from_hdf5('data/database_window13.h5', dataset="/y")
sample_weights = tfio.IODataset.from_hdf5('data/database_window13.h5', dataset="/sample_weight")

learn = tf.data.Dataset.zip((X, Y, sample_weights)).batch(1000).prefetch(tf.data.experimental.AUTOTUNE)

model = cnn.cnn()



history = model.fit(learn, epochs=30)

