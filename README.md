# 🦙 Sugar_pred 🦙

✍ Author:

**BELAKTIB Anas**
Master 2 Bio-informatics at *Univerité de Paris*.

# 🔎 Interesting path
- 📑 Report: `doc/report.pdf`
- 📢 Oral presentation: `doc/presentation`
- 🖥 Main: `main.py`

## 🤔 Context
Interactions between proteins and carbohydrates play a significant function in different biological processes. Recently, their role has been brought to the forefront due to the importance of carbohydrate modeling in understanding the replication of the SARS-CoV-2 virus1,2. Still, due to their complex nature and the impressive diversity of carbohydrates, accurate theoretical prediction tools modeling these interactions remain critically lacking. In light of this, we recovered experimental structures of protein-carbohydrate bound complexes from an existing database3 and curated our own protein-carbohydrate binding site database. Our team is currently designing various deep-learning architectures to predict the location of carbohydrate binding sites on a protein surface.
### 🐍 Conda environment

To use this program, you will need to create a conda environment like so:

```bash
mamba env create --file environement.yml
conda env create --file environement.yml
conda activate tensorflow-gpu-2.6
```

To launch this program, simply use the next commands (after the activation of the conda environment):

```bash
python3 main.py --path_to_your_dataset
```
### Dataset
Dataset must be a h5py File. This file need contains differents parts:
- "X_training"
- "Y_training"
- "X_validation"
- "Y_validation"
- "sample_weights_training"
- "sample_weights_training"

### Change or test hyperparameters
To Change or test hyperparameters you need to go at line : 235-237 of main.py and change as you want

### Defaut hyperparameters
batch_size = 64
EPOCHS = 20
learning_rate_list = [1e-4, 1e-5]  

### Change the model
To change the model you need go at line : 243 of main.py and change as you want
You can find the different  model in the folder /src.
list of different model avaible : cnn.cnn(lr), inception.inception(), gru.gru(), simple.simple(lr)
- 📑 NOTES : cnn.cnn(),simple.simple() need specify a learning rate argument lr => 1e-4 by default on keras

### Defaut Model
   model_cnn = cnn.cnn(lr)


