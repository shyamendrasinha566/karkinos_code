
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,roc_auc_score,roc_curve,classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

ml_file = pd.read_csv(r"C:\Users\Shyamendra Sinha\Desktop\R programming files\R PROGRAM\R OUTPUT FILES\ml_geneexp_model_file_uterine.csv")

print(ml_file.shape)

print(ml_file.columns)

# SEPARATE METADATA AND GENE FEATURE MATRIX 

metadata_col = ["Sample_ID", "Target"]

feature_col =[
    col for col in ml_file.columns
    if col not in metadata_col
]

metadata = ml_file[metadata_col]


# SET FEATURES

X = ml_file[feature_col]

# TARGET 

y = ml_file["Target"]


print("Features shape:", X.shape)
print("Target shape:", y.shape)

# preserve sample_ids 

sample_ids = ml_file["Sample_ID"]

# TRAIN/TEST SPLIT 

X_train, X_test,y_train,y_test,train_ids,test_ids = train_test_split(X,y,sample_ids, test_size=0.20,random_state=42,stratify=y)

print("Original dataset:", ml_file.shape)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# TRAINING DATA

train_df = X_train.copy()

train_df["Sample_ID"] = train_ids

train_df["Target"] = y_train

# TESTING DATA 

test_df = X_test.copy()

test_df["Sample_ID"] = test_ids

test_df["Target"] = y_test

train_df = train_df.reset_index(drop=True)

test_df = test_df.reset_index(drop=True)


# LABEL ENCODING 

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(ml_file["Target"])

print("Original classes:", label_encoder.classes_)

y_train = label_encoder.transform(train_df["Target"])

y_test = label_encoder.transform(test_df["Target"])

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# logsitic regression 

lr = LogisticRegression()
lr.fit(X_train,y_train)

y_pred_lr = lr.predict(X_test)

accuracy_lr = accuracy_score(y_test,y_pred_lr)

precision_lr = precision_score(y_test,y_pred_lr)

recall_lr = recall_score(y_test,y_pred_lr)

f1_lr = f1_score(y_test,y_pred_lr)

cm_lr = confusion_matrix(y_test,y_pred_lr)

print("LOGISTIC REGRESSION")

print(f"Accuracy_lr: {accuracy_lr * 100:.1f}%")
print(f"precision_lr : {precision_lr:.4f}")
print(f"recall_lr : {recall_lr:.4f}")
print(f"f1_lr : {f1_lr:.4f}")

print(cm_lr)


# XGBOOST 

xgb_model = xgb.XGBClassifier(n_estimators = 100, learning_rate =0.1, max_depth = 3, random_state = 42, n_jobs = 1)

xgb_model.fit(X_train,y_train)

y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test,y_pred_xgb)

precision_xgb = precision_score(y_test, y_pred_xgb)

recall_xgb = recall_score(y_test, y_pred_xgb)

f1_xgb = f1_score(y_test, y_pred_xgb)

cm_xgb = confusion_matrix(y_test, y_pred_xgb)


print("XGBOOST")

print(f"Accuracy : {accuracy_xgb * 100:.1f}%")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall   : {recall_xgb:.4f}")
print(f"F1-Score : {f1_xgb:.4f}")







