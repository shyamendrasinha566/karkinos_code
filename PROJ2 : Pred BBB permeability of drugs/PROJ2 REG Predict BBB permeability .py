
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error,mean_absolute_error

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


df = pd.read_csv("BBB_Permea dataset.csv")

df = df.drop(columns=["comments","threshold","group","Inchi","reference","CID"])


# Check validity of smiles 

valid_smiles = []
invalid_count = 0

for smi in df["SMILES"]:

    mol = Chem.MolFromSmiles(smi)

    if mol is not None:

        valid_smiles.append(True)
    else:
        valid_smiles.append(False)
        invalid_count += 1

print("Number of valid SMILES:", sum(valid_smiles))
print("Number of invalid SMILES:", invalid_count)

df = df[valid_smiles].copy()

df = df.dropna().reset_index(drop=True)


# DESCRIPTORS 

descriptor_names = [X[0] for X in Descriptors._descList]

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


def RDKit_descriptors(SMILES):

    molecular_descriptors= []

    for SMI in SMILES:

        mol = Chem.MolFromSmiles(SMI)

        desc = calculator.CalcDescriptors(mol)

        molecular_descriptors.append(desc)

    return molecular_descriptors

Mol_descriptors = RDKit_descriptors(df["SMILES"])

descriptors_df = pd.DataFrame(Mol_descriptors, columns=descriptor_names)

# COMBINE DESCRIPTOR LISTS 

df_new = pd.concat([df,descriptors_df], axis=1)

# SET FEATURE AND TARGET 

X = df_new[descriptor_names]

y = df_new["logBB"]



data = pd.concat([X, y], axis=1)

# Remove rows containing NaN

data = data.dropna().reset_index(drop=True)

# separate columns 

X = data.drop(columns=[y.name])
y = data[y.name]


# TRAIN/TEST 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# REMOVE CONSTANT DESCRIPTORS

from sklearn.feature_selection import VarianceThreshold

variance_selector = VarianceThreshold(threshold=0)

X_train_var = variance_selector.fit_transform(X_train)

X_test_var = variance_selector.transform(X_test)

selected_features = X_train.columns[variance_selector.get_support()]


# Convert back to DataFrame

X_train_var = pd.DataFrame(X_train_var, columns = selected_features, index=X_train.index)

X_test_var = pd.DataFrame(X_test_var, columns = selected_features, index=X_test.index)


# Remove Highly correlated Descirptors 

corr_matrix = X_train_var.corr().abs()


upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))

high_corr_features = [column
    for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.95)
]

print("Highly correlated features removed:",len(high_corr_features))


# FINAL EVALUATED TRAIN AND TEST FEATURES

X_train_Final = X_train_var.drop(columns=high_corr_features)

X_test_Final = X_test_var.drop(columns=high_corr_features)


# SINCE IPC VALUE WAS TOO LARGE FOR THE MODELS TO PROCESS 

X_train_Final = X_train_Final.drop(columns=["Ipc"])
X_test_Final = X_test_Final.drop(columns=["Ipc"])


# REGRESSION MODELS 


# LINEAR REGRESSION 

lr = LinearRegression()
lr.fit(X_train_Final,y_train)

y_pred_lr = lr.predict(X_test_Final)

r2_lr = r2_score(y_test,y_pred_lr)
mae_lr = mean_absolute_error(y_test,y_pred_lr)
mse_lr = mean_squared_error(y_test,y_pred_lr)


# EVALUATION

print("LINEAR REGRESSION")

print(f"R2 Score : {r2_lr:.4f}")
print(f"mae_lr : {mae_lr:.4f}")
print(f"mse_lr : {mse_lr:.4f}")


# XGBOOST 

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators= 100,learning_rate = 0.1,max_depth=3, random_state = 42,n_jobs =1)
xgb_model.fit(X_train_Final,y_train)

y_pred_xgb = xgb_model.predict(X_test_Final)

r2_xgb = r2_score(y_test,y_pred_xgb)
mae_xgb = mean_absolute_error(y_test,y_pred_xgb)
mse_xgb = mean_squared_error(y_test,y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

# RESULTS


print("XGBOOST")

print(f"R² : {r2_xgb:.4f}")
print(f"MAE: {mae_xgb:.4f}")
print(f"MSE : {mse_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.4f}")


# SUPPORT VECTOR REGRESSION (SVR)

from sklearn.preprocessing import StandardScaler

svr_model = SVR(kernel="rbf",C=100,gamma="scale",epsilon=0.1)

scaler = StandardScaler()

X_train_svr = scaler.fit_transform(X_train_Final)
X_test_svr = scaler.transform(X_test_Final)

svr_model.fit(X_train_svr,y_train)


svr_pred = svr_model.predict(X_test_svr)

r2_svr = r2_score(y_test,svr_pred)
mae_svr = mean_absolute_error(y_test,svr_pred)
mse_svr = mean_squared_error(y_test,svr_pred)

# EVALUATION

print("SVR")

print(f"r2_SVR : {r2_svr:.4f}")
print(f"mae_SVR : {mae_svr:.4f}")
print(f"mse_SVR : {mse_svr:.4f}")



# 5 FOLD CROSS VALIDATION 

from sklearn.model_selection import KFold,cross_val_score

kf = KFold(n_splits=5, shuffle=True,random_state=42)


def cross_validate_model(model,X,y,model_name):

    r2_scores = cross_val_score(model,X,y,cv=kf,scoring= "r2",n_jobs=1)

    mse_scores = cross_val_score(model,X,y,cv=kf,scoring="neg_mean_squared_error",n_jobs=1)

    mse_scores = -mse_scores
    
    rmse_scores = np.sqrt(mse_scores)

    mae_scores = cross_val_score(model,X,y,cv=kf,scoring="neg_mean_absolute_error",n_jobs=-1)

    mae_scores = -mae_scores

    # EVALUATE 

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

# CROSS VALIDATE 

cross_validate_model(xgb_model,X_train_Final,y_train,"XGBoost")
cross_validate_model(svr_model,X_train_Final,y_train,"SVR")
cross_validate_model(lr,X_train_Final,y_train,"LinearRegression")



# FROM THE ABOVE ANALYSIS OF THE OUTCOME DATA (XGBOOST) SHOWED BETTER RESULT AS A MODEL 

# XGBOOST HYPERPARAMETER OPTIMIZATION 


from sklearn.model_selection import RandomizedSearchCV

xgb_param_dist = {
    "n_estimators": [100, 200, 300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5, 1],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [0.1, 1, 5, 10]
}

xgb_base = xgb.XGBRegressor(random_state=42,n_jobs=1)

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_dist,
    n_iter=50,                 
    scoring="r2",
    cv=kf,                    
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_random_search.fit(X_train_Final,y_train)

print(xgb_random_search.best_params_)

print("Best CV R2:",round(xgb_random_search.best_score_,4))



# OPTIMIZED XGBOOST MODEL 

best_xgb_model = xgb_random_search.best_estimator_

y_pred_xgb_model = best_xgb_model.predict(X_test_Final)



r2_best_rf = r2_score(y_test,y_pred_xgb_model)

mse_best_rf = mean_squared_error(y_test,y_pred_xgb_model)

rmse_best_rf = np.sqrt(mse_best_rf)

mae_best_rf = mean_absolute_error(y_test,y_pred_xgb_model)


print("OPTIMIZED XGBOOST VALUES")

print(f"R2 Score : {r2_best_rf:.4f}")
print(f"MSE: {mse_best_rf:.4f}")
print(f"RMSE: {rmse_best_rf:.4f}")
print(f"MAE: {mae_best_rf:.4f}")


# SAVE THE MODEL 

import joblib

model_package = {
    "model": best_xgb_model,
    "selected_features": list(X_train_Final.columns),
    "descriptor_names": descriptor_names,
    "high_corr_features": high_corr_features
}


joblib.dump(model_package,"Optimized xgboost model.pkl")

print("MODEL SAVED")



# LOAD THE SAVE MODEL 

model_package = joblib.load("Optimized xgboost model.pkl")

best_xgb_model = model_package["model"]
selected_features = model_package["selected_features"]
descriptor_names = model_package["descriptor_names"]



# TO SEE WHICH FEATURE IS HAVING THE MOST INFLUENCE ON THE OUTPUT OF THIS MODEL 

# GLOBAL SHAP 

import shap

explainer = shap.TreeExplainer(best_xgb_model)

shap_values = explainer.shap_values(X_test_Final)

shap.summary_plot(shap_values,X_test_Final)


# PREDICT LOGBB VALUE OF A NEW DRUG MOLECULE (DIAZEPAM) 

new_drug = "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13"

mol = Chem.MolFromSmiles(new_drug)

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

descriptor = calculator.CalcDescriptors(mol)

new_drug_df = pd.DataFrame([descriptor],columns=descriptor_names)


# EVALUATE 

New_drug_final = new_drug_df[selected_features]

predicted = best_xgb_model.predict(New_drug_final)[0]


print("SMILES:",new_drug)

print(f"Predicted LogBB :{predicted:.4f}")