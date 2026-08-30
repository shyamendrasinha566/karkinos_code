

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.model_selection import (train_test_split,KFold,cross_val_score,RandomizedSearchCV)

from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (r2_score,mean_absolute_error,mean_squared_error)

import xgboost as xgb


# LOAD DATA

df = pd.read_csv("approved_drugs_clean.csv")

# Rename SMILES column

df = df.rename(columns={"SMILES": "smiles"})

# Remove missing SMILES

df = df.dropna(subset=["smiles"]).copy()


# SMILES VALIDITY

valid_smiles = []
invalid_count = 0

for smi in df["smiles"]:

    mol = Chem.MolFromSmiles(smi)

    if mol is not None:
        valid_smiles.append(True)
    else:
        valid_smiles.append(False)
        invalid_count += 1


# Keep only valid SMILES

df = df[valid_smiles].reset_index(drop=True)


# RDKit DESCRIPTORS

descriptor_names = [X[0] for X in Descriptors._descList]

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


def RDkit_descriptors(smiles):

    molecular_descriptor = []

    for smi in smiles:

        mol = Chem.MolFromSmiles(smi)

        descriptor = calculator.CalcDescriptors(mol)

        molecular_descriptor.append(descriptor)

    return molecular_descriptor


Mol_descriptors = RDkit_descriptors(df["smiles"])


# CONVERT DESCRIPTORS TO DATAFRAME

descriptor_df = pd.DataFrame(Mol_descriptors,columns=descriptor_names)

# Combine 

df_new = pd.concat([df, descriptor_df],axis=1)


# CHECK TARGET

print("\nMissing LOGS values:", df_new["LOGS"].isna().sum())


# Select descriptors + molecular weight



X = df_new [descriptor_names + ["MOLECULAR_WEIGHT"]]

y = df_new["LOGS"]


# Combine X and y

data = pd.concat([X, y], axis=1)

# Remove rows containing NaN

data = data.dropna().reset_index(drop=True)

X = data.drop(columns=[y.name])
y = data[y.name]


# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=42)



# REMOVE CONSTANT DESCRIPTORS

variance_selector = VarianceThreshold(threshold=0)

X_train_var = variance_selector.fit_transform(X_train)

X_test_var = variance_selector.transform(X_test)

selected_features = X_train.columns[variance_selector.get_support()]



# Convert back to DataFrame


X_train_var = pd.DataFrame(X_train_var, columns=selected_features, index=X_train.index)


X_test_var = pd.DataFrame(X_test_var, columns=selected_features, index=X_test.index)


# REMOVE HIGHLY CORRELATED DESCRIPTORS

corr_matrix = X_train_var.corr().abs()


upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))

high_corr_features = [
    column
    for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.95)
]

print("\nHighly correlated features removed:",len(high_corr_features))


# Remove correlated descriptors

X_train_final = X_train_var.drop(columns=high_corr_features)


X_test_final = X_test_var.drop(columns=high_corr_features)


# LINEAR REGRESSION

lr = LinearRegression()

lr.fit(X_train_final,y_train)

y_pred_lr = lr.predict(X_test_final)

# Evaluation

r2_lr = r2_score(y_test,y_pred_lr)
mse_lr = mean_squared_error(y_test,y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test,y_pred_lr)

print("LINEAR REGRESSION")

print(f"R2 Score : {r2_lr:.4f}")
print(f"MSE      : {mse_lr:.4f}")
print(f"RMSE     : {rmse_lr:.4f}")
print(f"MAE      : {mae_lr:.4f}")



# XGBOOST

xgb_model = xgb.XGBRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42,n_jobs=-1)

xgb_model.fit(X_train_final,y_train)

y_pred_xgb = xgb_model.predict(X_test_final)


# Evaluation

r2_xgb = r2_score(y_test,y_pred_xgb)

mse_xgb = mean_squared_error(y_test,y_pred_xgb)

rmse_xgb = np.sqrt(mse_xgb)

mae_xgb = mean_absolute_error(y_test,y_pred_xgb)

print("XCBOOST")

print(f"R2 Score : {r2_xgb:.4f}")
print(f"MSE : {mse_xgb:.4f}")
print(f"RMSE : {rmse_xgb:.4f}")
print(f"MAE : {mae_xgb:.4f}")



# RANDOM FOREST

rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)

rf_model.fit(X_train_final,y_train)
y_pred_rf = rf_model.predict(X_test_final)


# Evaluation

r2_rf = r2_score(y_test,y_pred_rf)

mse_rf = mean_squared_error(y_test,y_pred_rf)

rmse_rf = np.sqrt(mse_rf)

mae_rf = mean_absolute_error(y_test,y_pred_rf)

print("RANDOM FOREST")

print(f"R2 Score : {r2_rf:.4f}")
print(f"MSE : {mse_rf:.4f}")
print(f"RMSE : {rmse_rf:.4f}")
print(f"MAE : {mae_rf:.4f}")



# 5-FOLD CROSS VALIDATION


kf = KFold(n_splits=5, shuffle=True,random_state=42)


def cross_validate_model(model,X,y,model_name):


    r2_scores = cross_val_score(model,X,y,cv=kf,scoring="r2",n_jobs=-1)


    mse_scores = cross_val_score(model, X,y,cv=kf,scoring="neg_mean_squared_error",n_jobs=-1)

    mse_scores = -mse_scores

    rmse_scores = np.sqrt(mse_scores)

    # MAE

    mae_scores = cross_val_score(model,X,y,cv=kf,scoring="neg_mean_absolute_error",n_jobs=-1)

    # Convert negative MAE to positive
    mae_scores = -mae_scores


    print(
        f"R2   : {r2_scores.mean():.4f} "
        f"± {r2_scores.std():.4f}"
    )

    print(
        f"MSE  : {mse_scores.mean():.4f} "
        f"± {mse_scores.std():.4f}"
    )

    print(
        f"RMSE : {rmse_scores.mean():.4f} "
        f"± {rmse_scores.std():.4f}"
    )

    print(
        f"MAE  : {mae_scores.mean():.4f} "
        f"± {mae_scores.std():.4f}"
    )


# CROSS VALIDATE MODELS


cross_validate_model(xgb_model,X_train_final,y_train,"XGBOOST")


cross_validate_model(rf_model,X_train_final,y_train,"RANDOM FOREST")



# RANDOM FOREST HYPERPARAMETER OPTIMIZATION



rf_param_grid = {

    "n_estimators": [
        200,
        300,
        500,
        800
    ],

    "max_depth": [
        None,
        5,
        10,
        15,
        20
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2",
        0.5,
        0.8
    ]
}


rf_base = RandomForestRegressor(random_state=42,n_jobs=-1)


rf_search = RandomizedSearchCV(

    estimator=rf_base,

    param_distributions=rf_param_grid,

    n_iter=50,

    scoring="r2",

    cv=5,

    random_state=42,

    n_jobs=-1,

    verbose=2,

    return_train_score=True
)

rf_search.fit(X_train_final,y_train)


# BEST PARAMETERS

print(rf_search.best_params_)

print(Best CV R2:",round(rf_search.best_score_, 4))


#  OPTIMIZED RANDOM FOREST

best_rf = rf_search.best_estimator_


y_pred_best_rf = best_rf.predict( X_test_final)


r2_best_rf = r2_score(y_test,y_pred_best_rf)

mse_best_rf = mean_squared_error(y_test,y_pred_best_rf)

rmse_best_rf = np.sqrt( mse_best_rf)

mae_best_rf = mean_absolute_error(y_test,y_pred_best_rf)


print("OPTIMIZED RANDOM FOREST - TEST SET")


print(f"R2 Score : {r2_best_rf:.4f}")

print(f"MSE: {mse_best_rf:.4f}")

print(f"RMSE: {rmse_best_rf:.4f}")

print(f"MAE: {mae_best_rf:.4f}")


# BEST SELECTOR 

best_rf = rf_search.best_estimator_


# SELECT RANDOMIZED MODEL 

best_rf_pred = best_rf.predict(X_test_final)

r2 = r2_score(y_test,best_rf_pred)

mae = mean_absolute_error(y_test,best_rf_pred)

mse = mean_squared_error(y_test,best_rf_pred)

rmse = np.sqrt(mse)

print("\nOPTIMIZED RANDOM FOREST")

print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")

# SAVE THE MODEL 


import joblib

model_package = {
    "model": best_rf,
    "selected_features": list(X_train_final.columns),
    "descriptor_names": descriptor_names,
    "high_corr_features": high_corr_features
}


joblib.dump(model_package,"optimized_random_forest_logS_model.pkl")

print("\nModel saved successfully!")


# LOAD THE SAVED MODEL 

model_package = joblib.load("optimized_random_forest_logS_model.pkl")

best_rf = model_package["model"]
selected_features = model_package["selected_features"]
descriptor_names = model_package["descriptor_names"]


# NEW DRUG MOLECULE 

new_drug = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CC=C5"


mol = Chem.MolFromSmiles(new_drug)


calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

descriptors = calculator.CalcDescriptors(mol)


new_drug_df = pd.DataFrame([descriptors],columns=descriptor_names)

New_drug_final = new_drug_df[selected_features]

predicted = best_rf.predict(New_drug_final)


print("SMILES:", new_drug)

print(f"Predicted LogS : {predicted:.4f}")
