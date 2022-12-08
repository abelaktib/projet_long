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
    dense = Dense(100)(flat)
    drop4 = Dropout(0.5)(dense)
    dense2 = Dense(13)(drop4)
    dense3 = Dense(2, activation="softmax")(dense2)
    output = Flatten()(dense3)

    # Set the model.
    model = Model(inputs=inputs, outputs=output)
    

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    # Compile then return the model.
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  weighted_metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC(),
                           tf.keras.metrics.AUC(curve="PR"),"accuracy"])
    return model


# model = cnn()
# print(model.summary())

