"""Create a CNN neural network.
"""

__authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "03/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# [T]
import tensorflow as tf
import numpy as np

# [K]
from keras import Model, Input
from keras.layers import Dense, Conv1D, Dropout, Flatten, Reshape


def cnn(lr):
    """Apply a CNN to a given input.

    Parameters
    ----------
    inputs : keras.Input
        The input data.
    original : keras.Input
        The original input data.

    Returns
    -------
    Model.compile
        A compile model to be used for training.
    """
    # Neural network.
    inputs = Input(shape=(321, 13))
    reshape = Reshape((13, 321))(inputs)
    crop = tf.keras.layers.Cropping1D(cropping=6)(reshape)
    reshape2 = Reshape((321, 1))(crop)
    conv = Conv1D(filters=100, kernel_size=1, padding="same")(reshape2)
    
    drop = Dropout(0.4)(conv)

    conv2 = Conv1D(filters=int(100*(1.2**1)), kernel_size=3,
                   padding="same")(drop)
    drop2 = Dropout(0.4)(conv2)

    conv3 = Conv1D(filters=int(100*(1.2**2)), kernel_size=3,
                   padding="same")(drop2)
    drop3 = Dropout(0.4)(conv3)

    
    # Set the output.
    flat = Flatten()(drop3)
    dense = Dense(100)(flat)
    drop4 = Dropout(0.5)(dense)
    dense2 = Dense(13)(drop4)
    dense3 = Dense(2, activation="softmax")(dense2)
    output = Flatten()(dense3)

    # Set the model.
    model = Model(inputs=inputs, outputs=output)
    

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # Compile then return the model.
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  weighted_metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
                            tf.keras.metrics.Precision(name="precision"),
                            tf.keras.metrics.Recall(name="recall"),
                            tf.keras.metrics.AUC(name="auc"),
                            tf.keras.metrics.AUC(name="auc_1",curve="PR"),"accuracy"])
    return model


# model = cnn()
# print(model.summary())

