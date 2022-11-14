"""Google inception neural network.
"""

# [N]
import tensorflow as tf
# [K]
from keras import Input, Model
from keras.layers import Conv1D, Dense, concatenate, MaxPooling1D, Multiply,Flatten
from keras.layers import Dropout


__authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "10/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


def __inception(inputs):
    """Return a block of inception.
    Parameters
    ----------
    inputs : keras.Input
        The input data.
    Returns
    -------
    keras.layers
        a block of inception.
    """
    # First convolution.
    conv_1 = Conv1D(22, (1), padding="same", activation="relu")(inputs)
    conv_1 = Conv1D(22, (3), padding="same", activation="relu")(conv_1)
    conv_1 = Dropout(0.2)(conv_1)

    # Second convolution.
    conv_2 = Conv1D(22, (1), padding="same", activation="relu")(inputs)
    conv_2 = Conv1D(22, (5), padding="same", activation="relu")(conv_2)
    conv_2 = Dropout(0.2)(conv_2)

    # Third convolution.
    conv_3 = MaxPooling1D((3), strides=(1), padding="same")(inputs)
    conv_3 = Conv1D(22, (1), padding="same", activation="relu")(conv_3)
    conv_3 = Dropout(0.2)(conv_3)

    # Last convolution.
    conv_4 = Conv1D(22, (1), padding="same", activation="relu")(inputs)
    conv_4 = Dropout(0.2)(conv_4)

    conv = concatenate([conv_1, conv_2, conv_3, conv_4], axis=2)

    return conv


def inception(n_inception=2):
    """Apply a inception to a given input, with a given mask.
    Parameters
    ----------
    inputs : keras.Input
        The input data.
    n_inceptions : int
        Number of inceptions block.
    Returns
    -------
    Model.compile
        A compile model to be used for training.
    """
    print("=" * 80 + "\n")
    print(f"Doing a google inception neural network with {n_inception} "
          "modules.\n")
    print("=" * 80 + "\n")
    
    inputs = Input(shape=(321, 13))
    
  ### Inception layers.layer
    inception_i = inputs
    for _ in range(n_inception):
        inception_i = __inception(inception_i)  
                
    dense_1 = Dense(1200, activation="relu")(inception_i)
    drop_1 = Dropout(0.2)(dense_1)

    # Dense layers.
    dense_2 = Dense(600, activation="relu")(drop_1)
    drop_2 = Dropout(0.2)(dense_2)

    dense_3 = Dense(150, activation="relu")(drop_2)
    drop_3 = Dropout(0.2)(dense_3)
    flat_1 = Flatten() (drop_3)
    dense_4 = Dense(20, activation="relu")(flat_1)
    drop_4 = Dropout(0.2)(dense_4)

    output = Dense(13, activation="linear")(drop_4)

    # Set the model.
    model = Model(inputs=inputs, outputs=output)


    # Compile then return the model.
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=tf.keras.metrics.BinaryCrossentropy(from_logits=True), weighted_metrics=["accuracy"])


    return model


# if __name__ == "__main__":
#     model = inception()
#     print(model.summary())