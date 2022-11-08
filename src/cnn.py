"""Create a CNN neural network.
"""

__authors__ = ["BELAKTIB Anas"]
__contact__ = [ "anas.belaktib@etu.u-paris.fr"]
__date__ = "03/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# [T]
import tensorflow as tf
import numpy as np
# [K]
from keras import Model,Input
from keras.layers import Dense, Conv1D, Dropout, Flatten

def cnn():
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
    inputs = Input(shape=(512,320))
    conv = Conv1D(20, 320, activation="relu", kernel_initializer="he_uniform"
                    , padding="same")(inputs)
    drop = Dropout(0.2)(conv)
    conv2 = Conv1D(64, 320, activation="relu", kernel_initializer="he_uniform",
                    padding="same")(drop)
    drop2 = Dropout(0.2)(conv2)

    conv3 = Conv1D(64, 320, activation="relu", kernel_initializer="he_uniform",
                    padding="same")(drop2)
    drop3 = Dropout(0.2)(conv3)

    # Set the output.
    drop3 = Flatten()(drop3)
    output = Dense(512, activation="softmax")(drop3)

    # Set the model.
    model = Model(inputs=inputs, outputs=output)

    # Compile then return the model.
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'], weighted_metrics=["accuracy"])
    return model

# model = cnn()
# print(model.summary())