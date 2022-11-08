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


with open('data/esm2_8M_dict.p', "rb") as pickle_file:
    database_dict = pkl.load(pickle_file)  # open data

data = pd.DataFrame.from_dict({(i): database_dict[i]
                               for i in database_dict.keys()},
                              orient='index') # Transform nested dict into dataframe




x = data["x"] # Extract features from the dataframe

y = data["y"] # Extract Vector to predict from the dataframe

sample_weights= data["sample_weight"] # Extract sample weight

length = max(map(len, x)) # find the max len seq = 5126


matri = np.zeros((8196,5126,320), dtype=np.float16) # Create a zero matrice for inputs

to_pred = np.zeros((8196,5126), dtype=int) # Create a zero matrice for outputs


for i in range(x.shape[0]):
    seq_len = x[i].shape[0]
    matri[i][0:seq_len] = x[i]
    to_pred[i][0:seq_len] = y[i]


matri=matri[:,0:512,:]
# print(f"la taille est de {matri.shape}")
to_pred=to_pred[:,0:512]
# print(f"la taille est de {to_pred.shape}")


X=np.asarray(matri)


## Transform numpy into h5 file
h5f_x = h5py.File('/data/X.h5', 'w')
h5f_x.create_dataset('dataset_1', data=X)
h5f_x.close()


h5f_y = h5py.File('/data/to_pred.h5', 'w')
h5f_y.create_dataset('dataset_2', data=to_pred)
h5f_y.close()



# open new h5 file
with h5py.File("/data/X.h5", "r") as h5f_x:
    X_learn = h5f_x["x_val"][:]
    
with h5py.File("/data/to_pred.h5", "r") as h5f_y:
    Y_learn = h5f_y["y_val"][:]
    
#Calculate class weights
Y_tmp = np.argmax(Y_learn, axis=1)
class_weights_learn = class_weight.compute_class_weight("balanced", np.unique(Y_tmp), Y_tmp)
class_weights_learn = dict(enumerate(class_weights_learn))
sample_weights_learn = class_weight.compute_sample_weight(class_weights_learn, Y_tmp)
# Transform numpy array into tf.data.Dataset
sample_weights_learn = tf.data.Dataset.from_tensor_slices((sample_weights_learn))    
    
    

# Transform numpy array into tf.data.Dataset
X_learn = tfio.IODataset.from_hdf5(X_learn, dataset="/data/X.h5")
Y_learn = tfio.IODataset.from_hdf5(Y_learn, dataset="/data/to_pred.h5")


learn = tf.data.Dataset.zip((X_learn, Y_learn, sample_weights_learn)).batch(1000).prefetch(tf.data.experimental.AUTOTUNE)


# tensor1 = tf.convert_to_tensor(X)
# # # print(tensor1)

model = cnn.cnn()



history = model.fit(learn, validation_split=0.2,epochs=2, batch_size=50)
