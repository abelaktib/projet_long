# IMPORTS
import pickle as pkl
import pandas as pd
import numpy as np
import tensorflow as tf
import src.cnn as cnn


with open('data/esm2_8M_dict.p', "rb") as pickle_file:
    database_dict = pkl.load(pickle_file)  # open data

data = pd.DataFrame.from_dict({(i): database_dict[i]
                               for i in database_dict.keys()},
                              orient='index') # Transform nested dict into dataframe




x = data["x"] # Extract features from the dataframe

y = data["y"] # Extract Vector to predict from the dataframe

length = max(map(len, x)) # find the max len seq = 5126


matri = np.zeros((8196,5126,320), dtype=np.float16) # Create a zero matrice for inputs

to_pred = np.zeros((8196,5126), dtype=int) # Create a zero matrice for outputs

for i in range(x.shape[0]):
    seq_len = x[i].shape[0]
    matri[i][0:seq_len] = x[i]
    to_pred[i][0:seq_len] = y[i]

print(f"{matri[0][0:324]=}")
print(f"{x[0]=}")

X=np.asarray(matri)

# print(to_pred.shape)
# print(X.shape)

tensor1 = tf.convert_to_tensor(X)
# # print(tensor1)

model = cnn.cnn()

history = model.fit(tensor1, to_pred, validation_split=0.2,epochs=2, batch_size=50)
