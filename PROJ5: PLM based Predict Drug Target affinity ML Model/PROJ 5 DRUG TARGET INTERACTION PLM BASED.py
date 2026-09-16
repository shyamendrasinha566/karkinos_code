
import numpy as np
import pandas as pd

''''
df = pd.read_csv("bindingdb_highconf_pKd_clean.csv")

target_sequence = "MAQALPWLLLWMGAGVLPAHGTQHGIRLPLRSGLGGAPLGLRLPRETDEEPEEPGRRGSFVEMVDNLRGKSGQGYYVEMTVGSPPQTLNILVDTGSSNFAVGAAPHPFLHRYYQRQLSSTYRDLRKGVYVPYTQGKWEGELGTDLVSIPHGPNVTVRANIAAITESDKFFINGSNWEGILGLAYAEIARPDDSLEPFFDSLVKQTHVPNLFSLQLCGAGFPLNQSEVLASVGGSMIIGGIDHSLYTGSLWYTPIRREWYYEVIIVRVEINGQDLKMDCKEYNYDKSIVDSGTTNLRLPKKVFEAAVKSIKAASSTEKFPDGFWLGEQLVCWQAGTTPWNIFPVISLYLMGEVTNQSFRITILPQQYLRPVEDVATSQDDCYKFAISQSSTGTVMGAVIMEGFYVVFDRARKRIGFAVSACHVHDEFRTAAVEGPFVTLDMEDCGYNIPQTDESTLMTIAYVMAAICALFMLPLCLMVCQWRCLRCLRQQHDDFADDISLLK"

selected_rows = df[df["target_sequence"].astype(str).str.strip() == target_sequence]

print("Number of rows:", len(selected_rows))

selected_rows.to_csv("Alzheimer drug target interact.csv",index=False)

'''''
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

import xgboost as xgb
from xgboost import XGBRegressor


df = pd.read_csv("Alzheimer drug target interact.csv")

df = df.rename(columns={"compound_iso_smiles" : "SMILES"})

# CHECK VALIDITY OF SMILES 


valid_smiles = []
invalid_smiles = 0

for smi in df["SMILES"]:

    mol = Chem.MolFromSmiles(smi)

    if mol is not None:

        valid_smiles.append(True)
    else:
        valid_smiles.append(False)

        invalid_smiles += 1


print("Number of valid SMILES:", sum(valid_smiles))
print("Number of invalid SMILES:", invalid_smiles)

df = df[valid_smiles].copy()

df = df.dropna().reset_index(drop=True)

# MORGAN FINGERPRINT

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def Morgan_fingerprint(SMILES):

    Fingerprints = []

    for SMI in SMILES:

        mol = Chem.MolFromSmiles(SMI)

        arr = np.zeros((2048,), dtype=np.int8) 
 
        fp = generator.GetFingerprint(mol) 
 
        ConvertToNumpyArray(fp, arr) 
 
        Fingerprints.append(arr)

    return np.array(Fingerprints) 

# CONVERT TO DATAFRAME 

Morgan_FP = Morgan_fingerprint(df["SMILES"])

Fp_columns  = [f"Morgan_{i}" for i in range(2048)]

Fingerprint = pd.DataFrame(Morgan_FP,columns=Fp_columns)

# COMBINE FILE 

Final_df = pd.concat([df.reset_index(drop=True), Fingerprint.reset_index(drop=True)],axis=1)


# PROTEIN LANGUAGE MODEL (PLM) FOR TARGET SEQUENCE 

import torch
from transformers import AutoTokenizer, AutoModel
import esm


target_sequence = ("MAQALPWLLLWMGAGVLPAHGTQHGIRLPLRSGLGGAPLGLRLPRETDEEPEEPGRRGSFVEMVDNLRGKSGQGYYVEMTVGSPPQTLNILVDTGSSNFAVGAAPHPFLHRYYQRQLSSTYRDLRKGVYVPYTQGKWEGELGTDLVSIPHGPNVTVRANIAAITESDKFFINGSNWEGILGLAYAEIARPDDSLEPFFDSLVKQTHVPNLFSLQLCGAGFPLNQSEVLASVGGSMIIGGIDHSLYTGSLWYTPIRREWYYEVIIVRVEINGQDLKMDCKEYNYDKSIVDSGTTNLRLPKKVFEAAVKSIKAASSTEKFPDGFWLGEQLVCWQAGTTPWNIFPVISLYLMGEVTNQSFRITILPQQYLRPVEDVATSQDDCYKFAISQSSTGTVMGAVIMEGFYVVFDRARKRIGFAVSACHVHDEFRTAAVEGPFVTLDMEDCGYNIPQTDESTLMTIAYVMAAICALFMLPLCLMVCQWRCLRCLRQQHDDFADDISLLK")

target_sequence = "".join(target_sequence.split())

print(len(target_sequence))


# FIND THE MATCHED ROWS

matched_rows = df[df["target_sequence"] == target_sequence].copy()


# Load ESM-2

model_name = "facebook/esm2_t6_8M_UR50D"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()

# PROTEIN SEQUNECE INTO TOKEN 

inputs = tokenizer(target_sequence, return_tensors="pt")

# Generate PLM representation

with torch.no_grad():

    outputs = model(**inputs)

# Get amino-acid embeddings

embeddings = outputs.last_hidden_state

print("Embedding shape:")

print(embeddings.shape)

# Convert amino-acid embeddings into ONE protein embedding

protein_embedding = embeddings.mean(dim=1)

print(protein_embedding)

# CONVERT TENSOR TO NUMPY 

protein_embedding = (protein_embedding.detach().numpy())

# PROTEIN EMBEDDING FOR EACH DRUG 

number_of_drugs = len(Final_df)

Protein_features = np.repeat(protein_embedding,number_of_drugs,axis=0)

print(Protein_features.shape)


# CONCATANATING BOTH MORGAN FINGERPRINT AND PLM 

X = np.concatenate([Morgan_FP,Protein_features],axis=1)

y = Final_df["affinity"].astype(float).values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=42)


# XGBOOST 

xgb_model = XGBRegressor(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)

xgb_model.fit(X_train,y_train)
y_pred_xgb = xgb_model.predict(X_test)

R2_score_xgb = r2_score(y_test,y_pred_xgb)
MSE_xgb = mean_squared_error(y_test,y_pred_xgb)
RMSE_xgb = np.sqrt(MSE_xgb)
MAE_xgb = mean_absolute_error(y_test,y_pred_xgb)


print("XGBOOST")

print(f"R2_SCORE : {R2_score_xgb:.4f}")
print(f"MSE: {MSE_xgb:.4f}")
print(f"RMSE : {RMSE_xgb:.4f}")
print(f"MAE : {MAE_xgb:.4f}")


# SUPPORT VECTOR MACHINE (SVM)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

scaler = StandardScaler()

X_train_Final = scaler.fit_transform(X_train)
X_test_Final = scaler.transform(X_test)

svr_model = SVR(kernel="rbf",C=100, gamma="scale",epsilon=0.1)

svr_model.fit(X_train_Final,y_train)

y_pred_Final = svr_model.predict(X_test_Final)

r2_svr = r2_score(y_test,y_pred_Final)
mae_svr = mean_absolute_error(y_test,y_pred_Final)
mse_svr = mean_squared_error(y_test,y_pred_Final)

print("SVR")

print(f"R2_score_svr = {r2_svr:.4}")
print(f"mae_SVR : {mae_svr:.4f}")
print(f"mse_SVR : {mse_svr:.4f}")


# K NEAREST NEIGHBOURS (KNN)

from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(
    n_neighbors=5,
    weights="distance",
    metric="euclidean",
    n_jobs=-1
)
knn_model.fit(X_train_Final,y_train)

y_pred_knn = knn_model.predict(X_test_Final)

R2_score_knn = r2_score(y_test,y_pred_knn)
MSE_knn = mean_squared_error(y_test,y_pred_knn)
RMSE_knn = np.sqrt(MSE_knn)
MAE_knn = mean_absolute_error(y_test,y_pred_knn)


print("KNN")

print(f"R2_SCORE : {R2_score_knn:.4f}")

print(f"MSE: {MSE_knn:.4f}")

print(f"RMSE : {RMSE_knn:.4f}")

print(f"MAE : {MAE_knn:.4f}")


# 5 FOLD CROSS-VALIDATION 

from sklearn.model_selection import KFold,cross_val_score

kf = KFold(n_splits=5,shuffle=True, random_state=42)

def cross_validate_model(model,X,y,model_name):

    r2_scores = cross_val_score(model,X,y,scoring="r2",n_jobs=1)

    mse_scores = cross_val_score(model,X,y,cv=kf,scoring="neg_mean_squared_error",n_jobs=1)

    mse_scores = - mse_scores

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

# CROSS VALIDATE MODEL 

cross_validate_model(xgb_model,X_train,y_train,"XGBoost")
cross_validate_model(svr_model,X_train_Final,y_train,"SVR")
cross_validate_model(knn_model,X_train_Final,y_train,"KNN")


# HYPARAMETER OPTIMIZATION 

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

joblib.dump(best_xgb_model,"optimized_xgboost_PLM_Morgan.pkl")

print("Optimized XGBoost model saved successfully.")


# LOAD THE MODEL 

load_model = joblib.load("optimized_xgboost_PLM_Morgan.pkl")

# GRPAH PREDICTED DATA VS ACTUAL DATA

import matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(4,8))
plt.scatter(y_test, y_pred_xgb, alpha=0.7)

min_value = min(y_test.min(), y_pred_xgb_model.min())
max_value = max(y_test.max(), y_pred_xgb_model.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)


plt.xlabel("ACTUAL AFFINITY")
plt.ylabel("PREDICTED AFFINITY")
plt.title("PLOT BETWEEN ACTUAL VS PREDICTED PLOT - XGBOOST")

plt.tight_layout()
plt.show()












