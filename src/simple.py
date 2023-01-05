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


def simple(lr):
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
    conv = Conv1D(filters=int(100*(1.2**2)), kernel_size=3,
                   padding="same",activation="relu")(reshape2)
    drop = Dropout(0.1)(conv)

    # Set the output.
    flat = Flatten()(drop)
    dense3 = Dense(2, activation="softmax")(flat)
    # drop2 = Dropout(0.1)(dense3)
    output = Flatten()(dense3)

    # Set the model.
    model = Model(inputs=inputs, outputs=output)
    

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # Compile then return the model.
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  weighted_metrics=["accuracy",tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
                            tf.keras.metrics.Precision(name="precision"),
                            tf.keras.metrics.Recall(name="recall"),
                            tf.keras.metrics.AUC(name="auc"),
                            tf.keras.metrics.AUC(name="auc_1",curve="PR")])
    return model


# model = simple(1e-8)
# print(model.summary())

