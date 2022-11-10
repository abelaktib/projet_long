"""
Create a GRU neural network.
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
from keras.layers import Dense, Conv1D, Dropout, Flatten,Bidirectional,GRU
