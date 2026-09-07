

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

import numpy as np
import pandas as pd

import deepchem as dc
from rdkit import Chem

import matplotlib.pyplot as plt


# LOAD THE FILE 

df = pd.read_csv("BBB_Permea dataset.csv")

df = df.drop(columns=["comments", "group", "reference", "threshold", "Inchi"])

df = df.dropna(subset=["SMILES", "logBB"]).reset_index(drop=True)


# TO CHECK THE DATA RANGE 

print(df["logBB"].describe())


smiles = df["SMILES"].values

y = df["logBB"].astype(float).values


# Therefore reshape LogBB to one task.

y = y.reshape(-1, 1)


# MOLECULAR GRAPH FEATURIZATION

featurizer = dc.feat.ConvMolFeaturizer()

X_graph = featurizer.featurize(smiles)

# FEATURIZATION

valid_graphs  = []
valid_y = []
valid_smiles_final = []

for i, graph in enumerate(X_graph):

    if graph is not None:

        valid_graphs.append(graph)
        valid_y.append(y[i])
        valid_smiles_final.append(smiles[i])

X_graph = np.array(valid_graphs, dtype=object)
y = np.array(valid_y)
smiles = np.array(valid_smiles_final)



indices = np.arange(len(X_graph))

train_indices, test_indices = train_test_split(indices,test_size=0.20,random_state=42)


X_train = X_graph[train_indices]
X_test = X_graph[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

smiles_train = smiles[train_indices]
smiles_test = smiles[test_indices]


print("TRAINING DATA:", len(X_train))
print("TEST DATA:", len(X_test))


# DEEPCHEM DATASETS 

train_dataset = dc.data.NumpyDataset(X=X_train,y=y_train)

test_dataset = dc.data.NumpyDataset(X=X_test,y=y_test)

# CREATE GRAPH CONVOLUTIONAL MODEL 

model = dc.models.GraphConvModel(
    n_tasks=1,
    mode="regression",
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

    loss = model.fit(
        train_dataset,
        nb_epoch=1
    )

    loss_history.append(loss)

print(f"Epoch {epoch + 1:03d} | "f"Loss = {loss:.5f}")


# PREDICT

y_pred = model.predict(test_dataset)

y_pred = y_pred.reshape(-1)

y_test_flat = y_test.reshape(-1)


r2 = r2_score(y_test_flat,y_pred)
mae = mean_absolute_error(y_test_flat,y_pred)
mse = mean_squared_error(y_test_flat, y_pred)
rmse = np.sqrt(mse)


print(f"R² : {r2:.4f}")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


# RESIDUAL PLOT

residuals = (y_test_flat - y_pred)
plt.figure(figsize=(8, 5))
plt.scatter(y_pred,residuals,alpha=0.7)
plt.axhline(0,linestyle="--")
plt.xlabel("Predicted LogBB")
plt.ylabel("Residual")
plt.title("DeepChem LogBB Residual Plot")

plt.grid(True)
plt.tight_layout()
plt.show()

