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
from keras import Model, Input, optimizers
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
    inputs = Input(shape=(321, 13))
    conv = Conv1D(filters=100, kernel_size=1, padding="same")(inputs)
    drop = Dropout(0.2)(conv)

    conv2 = Conv1D(filters=int(100*(1.2**1)), kernel_size=3,
                   padding="same")(drop)
    drop2 = Dropout(0.2)(conv2)

    conv3 = Conv1D(filters=int(100*(1.2**2)), kernel_size=3,
                   padding="same")(drop2)
    drop3 = Dropout(0.2)(conv3)

    
    # Set the output.
    flat = Flatten()(drop3)
    dense = Dense(100,activation="softmax")(flat)
    drop4 = Dropout(0.2)(dense)
    dense2 = Dense(13,activation="softmax")(drop4)
    drop5 = Dropout(0.2)(dense2)
    dense3 = Dense(2, activation="softmax")(drop5)
    output = Flatten()(dense3)

    # Set the model.
    model = Model(inputs=inputs, outputs=output)
    

    # opt = optimizers.Adam(learning_rate=1e-6)
    # Compile then return the model.
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5), weighted_metrics=["accuracy"])
    return model


model = cnn()
print(model.summary())


def cnn2():
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
    inputs = Input(shape=(512, 320))
    conv = Conv1D(filters=100, kernel_size=1, input_shape=(
        512, 320), padding="same", activation="relu")(inputs)
    drop = Dropout(0.2)(conv)

    conv2 = Conv1D(filters=int(100*(1.2**1)), kernel_size=3,
                   padding="same", activation="relu")(drop)
    drop2 = Dropout(0.2)(conv2)

    conv3 = Conv1D(filters=int(100*(1.2**2)), kernel_size=3,
                   padding="same", activation="relu")(drop2)
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

# model = cnn2()
# print(model.summary())
