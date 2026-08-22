
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import pandas as pd
import numpy as np


# READ THE FILE 

df=pd.read_csv('Chemical Smiles.csv')

df = df.rename(columns={'SMILES Representation': 'smiles'})




# CREATE MOL DECSRIPTORS FROM SMILES 

descriptors_names = [ X[0] for X in Descriptors._descList]

# Calculator for Descirptors

Calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_names)

# CALCULATING MOLECULE DESCRIPTOR

def RDkit_descriptors(smiles):

    molecular_descriptor = []

    for smi in smiles:

        mol = Chem.MolFromSmiles(smi)

        descriptor = Calculator.CalcDescriptors(mol)

        molecular_descriptor.append(descriptor)

    return molecular_descriptor


# Calculate descriptors

Mol_descriptors = RDkit_descriptors(df["smiles"])

# Convert to DataFrame

descriptor_df = pd.DataFrame(
    Mol_descriptors,
    columns=descriptors_names
)

df_1 = pd.concat([df, descriptor_df], axis=1)


# MACHINE LEARNING 

import sklearn
from sklearn.linear_model import LinearRegression,LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error


# FEATURE AND TARGET

X = descriptor_df[descriptors_names]
y = df["lgK"]


# Check NaN

print(X.isna().sum().sort_values(ascending=False))

# Drop columns containing NaN

X = X.dropna(axis=1)

print("Remaining features:", X.shape)


# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL

cl = LinearRegression()

cl.fit(X_train, y_train)

# PREDICTION

y_pred = cl.predict(X_test)

# EVALUATION

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R2 Score : {r2:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

