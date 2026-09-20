
import scanpy as sc
import numpy as np
import pandas as pd
import os
from scipy.io import mmread


# PRE-TREATMENT

pre_path = r"C:\Users\Shyamendra Sinha\Desktop\R programming files\scRNA FILES\GSE306339_RAW\PRE TREATMENT"

pre_matrix = mmread(
    pre_path + r"\GSM9197630_AML-A15-pre_matrix.mtx.gz"
).tocsr()

pre_barcodes = pd.read_csv(
    pre_path + r"\GSM9197630_AML-A15-pre_barcodes.tsv.gz",
    sep="\t",
    header=None
)

pre_features = pd.read_csv(
    pre_path + r"\GSM9197630_AML-A15-pre_features.tsv.gz",
    sep="\t",
    header=None
)

# Create AnnData

adata_pre = sc.AnnData(
    X=pre_matrix.T
)

# Cell names
adata_pre.obs_names = pre_barcodes.iloc[:, 0].astype(str).values

# Gene names
adata_pre.var_names = pre_features.iloc[:, 1].astype(str).values

# Make gene names unique
adata_pre.var_names_make_unique()


# Add metadata
adata_pre.obs["treatment"] = "Pre"
adata_pre.obs["sample"] = "AML-A15"


# print("PRE-TREATMENT")
# print(adata_pre)

# print("Matrix shape:", pre_matrix.shape)
# print("Cells:", adata_pre.n_obs)
# print("Genes:", adata_pre.n_vars)


# LOAD POST TREATMENT DATA 


post_path = r"C:\Users\Shyamendra Sinha\Desktop\R programming files\scRNA FILES\GSE306339_RAW\POST TREATMENT"

post_matrix = mmread(
    post_path + r"\GSM9197631_AML-A15-post_matrix.mtx.gz"
).tocsr()

post_barcodes = pd.read_csv(
    post_path + r"\GSM9197631_AML-A15-post_barcodes.tsv.gz",
    sep="\t",
    header=None
)

post_features = pd.read_csv(
    post_path + r"\GSM9197631_AML-A15-post_features.tsv.gz",
    sep="\t",
    header=None
)


adata_post = sc.AnnData(
    X=post_matrix.T
)

adata_post.obs_names = post_barcodes.iloc[:, 0].astype(str).values

adata_post.var_names = post_features.iloc[:, 1].astype(str).values

adata_post.var_names_make_unique()

adata_post.obs["treatment"] = "Post"
adata_post.obs["sample"] = "AML-A15"


# print("POST-TREATMENT")
# print(adata_post)

# print("Matrix shape:", post_matrix.shape)
# print("Cells:", adata_post.n_obs)
# print("Genes:", adata_post.n_vars)


# PRE-TREATED CELLS

# Calculate Mitochondrial genes 

adata_pre.var["mt"] = adata_pre.var_names.str.startswith("MT-")

print("NUMBER OF MITOCHONDRIAL GENES:", adata_pre.var["mt"].sum())

# QUALITY CONTROL 

sc.pp.calculate_qc_metrics(
    adata_pre,
    qc_vars=["mt"],
    inplace=True
)

print(adata_pre.obs[
         ["total_counts","n_genes_by_counts","pct_counts_mt"]
    ]
)

# Violoin Plot 

sc.pl.violin(
    adata_pre,
    ["total_counts","n_genes_by_counts","pct_counts_mt"],
    jitter = 0.4,
    multi_panel = True,
)

# FILTER DATA 

adata_pre = adata_pre[adata_pre.obs["n_genes_by_counts"] > 200].copy()

sc.pp.filter_genes(adata_pre, min_cells=3)

print("NUMBER OF CELLS AFTER QC:",  adata_pre.n_obs)
print("NUMBER OF GENES AFTER QC:",  adata_pre.n_vars)


# POST TREATED CELLS

# Calculate Mitochondrial genes 

adata_post.var["mt"] = adata_post.var_names.str.startswith("MT-")


print("NUMBER OF MITOCHONDRIAL GENES:", adata_post.var["mt"].sum())

# QUALITY CONTROL 

sc.pp.calculate_qc_metrics(
    adata_post,
    qc_vars=["mt"],
    inplace=True
)

print(adata_post.obs[
         ["total_counts","n_genes_by_counts","pct_counts_mt"]
    ]
)

# Violoin Plot 

sc.pl.violin(
    adata_post,
    ["total_counts","n_genes_by_counts","pct_counts_mt"],
    jitter = 0.4,
    multi_panel = True,
)

# FILTER DATA 

adata_post = adata_post[adata_post.obs["n_genes_by_counts"] > 200].copy()

sc.pp.filter_genes(adata_post, min_cells=3)

# SAVE RAW COUNTS
print("NUMBER OF CELLS AFTER QC:",  adata_post.n_obs)
print("NUMBER OF GENES AFTER QC:",  adata_post.n_vars)

adata_pre.layers["counts"] = adata_pre.X.copy()
adata_post.layers["counts"] = adata_post.X.copy()

# NORMALIZATION 

sc.pp.normalize_total(adata_post, target_sum = 1e4)
sc.pp.normalize_total(adata_pre, target_sum = 1e4)

# LOG TRANSFORMATION 

sc.pp.log1p(adata_pre)
sc.pp.log1p(adata_post)

# HIGH VARIBALE GENES 

sc.pp.highly_variable_genes(
    adata_pre,
    n_top_genes=3000,
    flavor="seurat"
)

sc.pp.highly_variable_genes(
    adata_post,
    n_top_genes=3000,
    flavor="seurat"
)

print("Number of HVGs:")

print(
    "PRE :",
    adata_pre.var["highly_variable"].sum()
)

print(
    "POST:",
    adata_post.var["highly_variable"].sum()
)

# COMBINE DATA 

adata = sc.concat(
    {
        "Pre": adata_pre,
        "Post": adata_post
    },
    join="outer",
    label="condition",
    index_unique="-"
)


print("CONDITION")

print(adata.obs["condition"].value_counts())

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    flavor="seurat",
    batch_key="condition"
)

print("After selecting HVGs:")

print(adata)

# SCALING 

adata.raw = adata.copy()

sc.pp.scale(adata, max_value=10)

# PCA

sc.pp.pca(adata, n_comps=50, svd_solver="arpack")

#sc.pl.pca(adata)
          
# NEIGHBOUR 

#sc.pl.pca_variance_ratio(adata, log=True)

sc.pp.neighbors(adata,n_neighbors=10, n_pcs=30)

# UMAP

sc.tl.umap(adata)

#sc.pl.umap(adata)

# LEIDEN 

sc.tl.leiden(adata, resolution=0.5)

sc.pl.umap(adata, color =["leiden"])

# MARKER GENES 

sc.tl.rank_genes_groups(
    adata,
    groupby="leiden",
    method="wilcoxon"
)

sc.pl.rank_genes_groups(
    adata,
    n_genes=10,
    sharey=False
)

markers = sc.get.rank_genes_groups_df(
    adata,
    group=None
)

print(markers)



# CELL TYPE ANNOTATION 
# IDENTIFY GENES PRESENT IN EACH CLUSTER AND CORESSPONDING CELL-TYPE 


celltype_map = {
    "0": "CD8_T_cell",
    "1": "Naive_T_cell",
    "2": "Unknown_Neuronal_like",
    "3": "Ribosomal_high_low_quality",
    "4": "Memory_T_cell",
    "5": "Classical_Monocyte",
    "6": "Mito_high_low_quality",
    "7": "NK_cell_CD16",
    "8": "B_cell",
    "9": "Progenitor_CD34",
    "10": "GZMK_Effector_T_cell",
    "11": "NK_cell_CD56bright",
    "12": "Megakaryocyte_precursor",
    "13": "Platelet",
    "14": "Plasma_cell",
    "15": "Ribosomal_high_low_quality",
    "16": "Gamma_delta_T_cell",
    "17": "Erythroid",
    "18": "Classical_Monocyte",
    "19": "Mito_high_low_quality",
    "20": "Mast_cell",
    "21": "B_cell",
    "22": "Fibroblast_stromal"
}

adata.obs["cell_type"] = (
    adata.obs["leiden"].map(celltype_map)
)

# CELL COUNTS PRE AND POST

cell_counts = pd.crosstab(
    adata.obs["condition"],
    adata.obs["cell_type"]
)

print("CELL COUNTS PRE vs POST:")
print(cell_counts)



# PRE AND POST CELL PERCENTAGES 

cell_percentages = (
    pd.crosstab(
        adata.obs["condition"],
        adata.obs["cell_type"],
        normalize="index"
    ) * 100
)

print("CELL PERCENTAGES PRE vs POST:")
print(cell_percentages)
