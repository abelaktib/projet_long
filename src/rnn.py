"""RNN
"""

# [N]
import tensorflow as tf
# [K]
from keras import Input, Model
from keras.layers import Conv1D, Dense, concatenate, MaxPooling1D, Multiply,Flatten,Bidirectional,GRU
from keras.layers import Dropout

__authors__ = ["BELAKTIB Anas"]
__contact__ = ["anas.belaktib@etu.u-paris.fr"]
__date__ = "10/11/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

def rnn():
    inputs = Input(shape=(321, 13))
    bi = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(inputs)
    bi = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(bi)
    bi = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(bi)
    flat_1 = Flatten() (bi)

    
    output = Dense(13, activation="linear")(flat_1)

    # Set the model.
    model = Model(inputs=inputs, outputs=output)
    
        # Compile then return the model.
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=tf.keras.metrics.BinaryCrossentropy(from_logits=True), weighted_metrics=["accuracy"])
    return model

# if __name__ == "__main__":
#     model = rnn()
#     print(model.summary())
