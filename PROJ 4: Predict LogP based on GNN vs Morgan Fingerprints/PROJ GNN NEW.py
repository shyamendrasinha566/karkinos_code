

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

import numpy as np
import pandas as pd

import deepchem as dc
from rdkit import Chem

import matplotlib.pyplot as plt

df = pd.read_csv("SMILES_IC50_logP_Data_Set.csv")

df = df.dropna(subset=["SMILES", "logP"]).reset_index(drop=True)

# TO CHECK THE DATA RANGE 

print(df["logP"].describe())

smiles = df["SMILES"].values

y = df["logP"].astype(float).values

y = y.reshape(-1,1)

# MOLECULAR FEATURIZATION 

featurizer = dc.feat.ConvMolFeaturizer()

X_graph = featurizer.featurize(smiles)

# FEATURIZER 

valid_graph = []
valid_y = []
valid_smiles_final = []

for i, graph in enumerate(X_graph):

    if graph is not None:

        valid_graph.append(graph)
        valid_y.append(y[i])
        valid_smiles_final.append(smiles[i])

X_graph = np.array(valid_graph, dtype=object)
y = np.array(valid_y)
smiles = np.array(valid_smiles_final)


indices = np.arange(len(X_graph))

train_indices, test_indices = train_test_split(indices,test_size=0.20,random_state=42)

X_train = X_graph[train_indices]
X_test = X_graph[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]


# DATASETS 

train_dataset = dc.data.NumpyDataset(X=X_train,y=y_train)
test_dataset = dc.data.NumpyDataset(X=X_test,y=y_test)

# CREATE GRAPH CONVOLUTIONAL MODEL 

model = dc.models.GraphConvModel(
        n_tasks = 1,
        mode = "regression",
        graph_conv_layers=[64, 64],
        dense_layer_size=128,
        dropout=0.20,
        batch_size=32,
        batch_normalize=False,
        model_dir="deepchem_logbb_model"
)


# TRAIN THE MODEL 

loss_history = []

for epoch in range(50):

    loss = model.fit(train_dataset,nb_epoch=1)

    loss_history.append(loss)

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Loss = {loss:.5f}"
    )


# PREDICT 

y_pred = model.predict(test_dataset)

y_pred = y_pred.reshape(-1)
y_test_flat = y_test.reshape(-1)


r2 = r2_score(y_test_flat,y_pred)
mae = mean_absolute_error(y_test_flat,y_pred)
mse = mean_squared_error(y_test_flat,y_pred)
rmse = np.sqrt(mse)


print(f"R² : {r2:.4f}")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")



# PLOT ACTUAL VS PREDICTED LOGP VALUE

plt.scatter(y_test_flat, y_pred)

plt.xlabel("Actual logP")
plt.ylabel("Predicted logP")
plt.title("Actual vs Predicted logP")

minimum = min(y_test_flat.min(), y_pred.min())
maximum = max(y_test_flat.max(), y_pred.max())

plt.plot([minimum, maximum],[minimum, maximum],linestyle="--")
plt.show()



# USING MORGAN FINGERPRINT

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

from xgboost import XGBRegressor


Descriptors_names = [X[0] for X in Descriptors._descList]

# generator 

Calculator = MoleculeDescriptors.MolecularDescriptorCalculator(Descriptors_names) 

def RDkit_descriptors(SMILES):

    Descriptors = []

    for smi in smiles:

        mol = Chem.MolFromSmiles(smi) 

        fp = Calculator.CalcDescriptors(mol)

        Descriptors.append(fp)

    return Descriptors, Descriptors_names


MOL_descriptors, desc_names = RDkit_descriptors(df["SMILES"])

# convert to dataframe 

descriptor_df = pd.DataFrame(MOL_descriptors, columns=desc_names)




def Morgan_fingerprints(smiles, radius=2, nBits=2048):

    fingerprints = []

# Generate Morgan fingerprint

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=nBits)

    for smi in smiles:

        mol = Chem.MolFromSmiles(smi)

        fp = generator.GetFingerprint(mol)

        arr = np.array(mol)

        ConvertToNumpyArray(fp,arr)

        fingerprints.append(fp)

    return np.array(fingerprints) 



# generate fingerprint

Morgan_fp = Morgan_fingerprints(df["SMILES"])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Morgan_fp,y,test_size=0.20,random_state=42)


model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


print(f"R² : {r2:.4f}")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")

plt.figure(figsize=(7,6))

plt.scatter(y_test,y_pred,alpha=0.7)


# Perfect prediction line

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot([minimum, maximum],[minimum, maximum],linestyle="--")


plt.xlabel("Actual logP")
plt.ylabel("Predicted logP")
plt.title("XGBoost: Actual vs Predicted logP")
plt.show()