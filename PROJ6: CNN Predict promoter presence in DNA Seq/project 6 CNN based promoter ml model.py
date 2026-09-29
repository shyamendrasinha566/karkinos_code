
import matplotlib.pyplot as plt 
import numpy as np

import scanpy as sc
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,classification_report,confusion_matrix,roc_auc_score 

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

print("TensorFlow Keras import successful")

df = pd.read_csv("promoter_binary_classification_dataset.csv")

print(df.shape)

print(df["label"].value_counts())


# ONE HOT ENCODING 

base_to_vector = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1]
}

def one_hot_encode(sequence):

    encoded = []

    for base in sequence:

        encoded.append(base_to_vector[base])

    return np.array(encoded,dtype=np.float32)

# CONVERT ALL SEQUENCES 

X = np.array([
    one_hot_encode(seq)
    for seq in df["sequence"]
])

y = df["label"].values.astype(np.float32)

common_length = X.shape[1]

# TRAIN/TEST SPLIT 

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=42,test_size=0.20, stratify=y)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# VALIDATION SPLIT 

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.20,random_state=42,stratify=y_train)

print("Final training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# BUILD A CNN MODEL  
 
model = tf.keras.Sequential([ 
 
    # Input 
    tf.keras.layers.Input( 
        shape=(common_length, 4) 
    ), 
 
    tf.keras.layers.Conv1D( 
        filters=128, 
        kernel_size=15, 
        activation="relu" 
    ), 
 
    tf.keras.layers.BatchNormalization(), 
 
    tf.keras.layers.MaxPooling1D( 
        pool_size = 2 
    ), 
 
# second convultion  
 
    tf.keras.layers.Conv1D( 
        filters=64, 
        kernel_size=10, 
        activation="relu" 
    ), 
 
    tf.keras.layers.BatchNormalization(), 
 
    tf.keras.layers.MaxPooling1D( 
        pool_size = 2 
    ), 
 
# Third convolution 
    
    tf.keras.layers.Conv1D( 
        filters=32, 
        kernel_size=5, 
        activation="relu" 
    ), 
 
# Dropout 
 
    tf.keras.layers.Dropout(0.3), 
 
# CONVERT FEATURES MAP TO VECTOR  
 
tf.keras.layers.GlobalMaxPooling1D(), 
 
    # Fully connected layer 
    tf.keras.layers.Dense( 
        64, 
        activation="relu" 
    ), 
 
    tf.keras.layers.Dropout(0.4), 
 
    # Binary classification 
    tf.keras.layers.Dense( 
        1, 
        activation="sigmoid" 
    ) 
]) 
 
# COMPILE MODEL  
 
model.compile( 
 
    optimizer= tf.keras.optimizers.Adam( 
        learning_rate=0.001 
    ), 
 
  loss="binary_crossentropy", 
 
    metrics=[ 
        "accuracy", 
        tf.keras.metrics.AUC(name="auc") 
    ] 
) 
 
model.summary()

# EARLY STOPPING 

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True
)


# TRAIN MODEL 

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
),

epochs=50,

batch_size=32,

callbacks= [
    early_stopping
],

verbose =1,
)

# TEST MODEL 

test_loss, test_accuracy, test_auc = model.evaluate(
    X_test,
    y_test,
    verbose = 0
)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Test AUC:", test_auc)


# PREDICTIONS 

y_probability = model.predict(
    X_test
).flatten()


y_prediction = (
    y_probability >= 0.5
).astype(int)

# CLASSIFICATION REPORT

print(classification_report(y_test,y_prediction,target_names=["Non-promoter", "Promoter"]))

# ROC-AUC 

roc_auc = roc_auc_score(y_test,y_prediction)

print("ROC-AUC SCORE :", roc_auc)

# CONFUSION MATRIX, RECALL AND F1_SCORE

cm = confusion_matrix(y_test,y_prediction)

F1_score_result = f1_score(y_test, y_prediction)

RECALL = recall_score(y_test, y_prediction)


print(f"f1_score: {F1_score_result:.4f}")

print(f"recall_score: {RECALL:.4f}")

print(cm)

# SAVE THE MODEL 

model.save("promoter_CNN_model.keras")

print("Model saved")
