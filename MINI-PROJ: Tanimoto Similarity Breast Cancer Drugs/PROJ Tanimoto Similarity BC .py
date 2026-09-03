
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.DataStructs import ConvertToNumpyArray

import matplotlib.pyplot as plt 

# LOAD THE FILE 

file = pd.read_csv("breast_cancer_SMILES.csv")

# CHECK VALIDITY OF SMILES 

valid_smiles = []
invalid_smiles = 0

for smi in file["SMILES"]:

    mol = Chem.MolFromSmiles(smi)

    if mol is not None:

        valid_smiles.append(True)
    else:
        valid_smiles.append(False)

        invalid_smiles += 1


print("Number of valid SMILES:", sum(valid_smiles))
print("Number of invalid SMILES:", invalid_smiles)



# FINGERPRINT GENERATION 

FP_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def Morgan_fingerprint(smiles): 
 
    fingerprints = [] 
 
    for smi in smiles: 
 
        mol = Chem.MolFromSmiles(smi) 
 
        arr = np.zeros((2048,), dtype=np.int8) 
 
        fp = FP_generator.GetFingerprint(mol) 
 
        ConvertToNumpyArray(fp, arr) 
 
        fingerprints.append(arr) 
 
    return np.array(fingerprints) 

# CONVERTING INTO DATAFRAME 

Morgan_Fp = Morgan_fingerprint(file["SMILES"])

Fp_columns  = [f"Morgan_{i}" for i in range(2048)]

Fingerprint = pd.DataFrame(Morgan_Fp,columns=Fp_columns)


# COMBINE BOTH THE FILE

Final = pd.concat([file.reset_index(drop=True), Fingerprint.reset_index(drop=True)],axis=1) 

# GENERATING FP FOR THE BREAST CANCER DRUG MOELCULE (PACLITAXEL)

new_drug_paclitaxel = "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"

new_mol = Chem.MolFromSmiles(new_drug_paclitaxel)

fp_new = FP_generator.GetFingerprint(new_mol)

# CALCULATING THE STRUCTURAL SIMILARIRTY USING TANIMOTO SIMILARITY 

similarities = []

for smi in file["SMILES"]:

    # Convert CURRENT SMILES into a molecule
    mol = Chem.MolFromSmiles(smi)

    if mol is not None:

        # Generate fingerprint for CURRENT drug
        fp = FP_generator.GetFingerprint(mol)

        # Calculate similarity with vepdegestrant
        similarity = TanimotoSimilarity(fp_new, fp)

        similarities.append(similarity)

    else:

        similarities.append(np.nan)


# Adding similarity to dataframe

Final["Tanimoto_paclitaxel"] = similarities

similar_drug_list = Final.sort_values("Tanimoto_paclitaxel",ascending=False)

print(similar_drug_list)

# PLOT GRAPH TO VISUALIZE WHICH IS MORE REFERENCE MOLECULE IS MORE SIMILAR TO THE TARGET 

# USING MATPLOTLIB 

plot_data = Final.sort_values("Tanimoto_paclitaxel",ascending=True)

plt.figure(figsize=(8, 10))

plt.barh(plot_data["Drug_Name"],plot_data["Tanimoto_paclitaxel"])
plt.xlabel("Tanimoto Similarity")
plt.ylabel("Drug Names")

plt.title("Structural Similarity against Paclitaxel")
plt.tight_layout()
plt.show()

