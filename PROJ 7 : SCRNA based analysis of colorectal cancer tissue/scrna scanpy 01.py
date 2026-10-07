
# import scanpy as sc
# import pandas as pd
# import numpy as np
# import os
# from scipy.io import mmread


# file_path = r"C:\Users\Shyamendra Sinha\Desktop\R programming files\scRNA FILES\GSE237726_RAW"

# pre_matrix = mmread(
#     os.path.join(
#         file_path,
#         "GSM7646208_BT-549_matrix.mtx.gz"
#     )
# ).tocsr()


# pre_barcodes = pd.read_csv(
#     os.path.join(
#         file_path,
#         "GSM7646208_BT-549_barcodes.tsv.gz"
#     ),
#     sep="\t",
#     header=None
# )


# pre_features = pd.read_csv(
#     os.path.join(
#         file_path,
#         "GSM7646208_BT-549_genes.tsv.gz"
#     ),
#     sep="\t",
#     header=None
# )


# print("Matrix shape:", pre_matrix.shape)
# print("Number of barcodes:", len(pre_barcodes))
# print("Number of genes:", len(pre_features))

# # Create AnnData


# adata = sc.AnnData(X=pre_matrix.T)


# adata.obs_names = pre_barcodes.iloc[:, 0].astype(str).values

# adata.var_names = pre_features.iloc[:, 1].astype(str).values

# adata.var_names_make_unique()

# print(adata)

# print("Number of cells:", adata.n_obs)
# print("Number of genes:", adata.n_vars)


# # CALCULAGE MITOCHODNTIAL GENES 

# adata.var["mt"] = adata.var_names.str.startswith("MT-")

# print("number of mitochondrial genes:", adata.var["mt"].sum())

# # QUALITY CONTROL 

# sc.pp.calculate_qc_metrics(
#     adata,
#     qc_vars=["mt"],
#     inplace=True
# )


# print(adata.obs[
#         ["total_counts","n_genes_by_counts","pct_counts_mt"]
#     ]
# )

# # VIOLIN PLOT 

# # sc.pl.violin(
# #     adata,
# #     ["total_counts","n_genes_by_counts","pct_counts_mt"],
# #     jitter=0.4,
# #     multi_panel=True
# # )

# # FILTER GENES 

# adata = adata[adata.obs["n_genes_by_counts"] > 200].copy()

# sc.pp.filter_genes(adata, min_cells=3)

# print("NUMBER OF CELLS AFTER QC:",  adata.n_obs)
# print("NUMBER OF GENES AFTER QC:",  adata.n_vars)

# # SAVE RAW COUNT 

# adata.layers["counts"] = adata.X.copy()

# # NORMALIZATION 

# sc.pp.normalize_total(adata, target_sum =1e4)

# # LOG TRANSFORM 

# sc.pp.log1p(adata)

# # HIGH VARIBALE GENES

# sc.pp.highly_variable_genes(
#     adata,
#     n_top_genes = 3000,
#     flavor = "seurat"
# )

# print("NUMBER OF HIGH VARIBALE GENES:", adata.var["highly_variable"].sum())


# # SCALING

# adata.raw = adata.copy()

# sc.pp.scale(adata, max_value=10)

# # PCA 

# sc.pp.pca(adata, n_comps=50, svd_solver = "arpack")

# sc.pl.pca(adata)

# # NEIGHBOUR 

# sc.pl.pca_variance_ratio(adata, log=True)

# sc.pp.neighbors(adata,n_neighbors=10, n_pcs=30)

# # UMAP 

# sc.tl.umap(adata)

# # LEIDEN 

# sc.tl.leiden(adata, resolution=0.5)

# sc.pl.umap(adata, color=["leiden"])


# # MARKER GENES 

# sc.tl.rank_genes_groups(
#     adata,
#     groupby="leiden",
#     method="wilcoxon"
# )

# sc.pl.rank_genes_groups(
#     adata,
#     n_genes=10,
#     sharey=False
# )

# markers = sc.get.rank_genes_groups_df(
#     adata,
#     group=None
# )

# print(markers)

# markers_0 = sc.get.rank_genes_groups_df(
#     adata,
#     group="0"
# )

# print(markers_0.head(20))

# sc.pl.rank_genes_groups_dotplot(
#     adata,
#     groups=["0"],
#     n_genes=20
# )


#########    COLORECTAL CANCER TISSUE WITHOUT TREATMENT ##########

import scanpy as sc
import numpy as np
import pandas as pd
import os
from scipy.io import mmread
from scipy import sparse

file_path = r"C:\Users\Shyamendra Sinha\Downloads\GSE330797_RAW"

file_matrix = mmread(
    file_path + r"\GSM9732960_3T-5GE_matrix.mtx.gz"
).tocsr()

print("Original matrix:", file_matrix.shape)
print("Original type:", type(file_matrix))




# # IMPORTANT: convert to CSR sparse matrix

# file_matrix = sparse.csr_matrix(file_matrix)

# print("After CSR conversion:", type(file_matrix))


file_barcodes = pd.read_csv(
    file_path + r"\GSM9732960_3T-5GE_barcodes.tsv.gz",
    sep="\t",
    header=None
)

file_features = pd.read_csv(
    file_path + r"\GSM9732960_3T-5GE_features.tsv.gz",
    sep="\t",
    header=None
)

print(file_features.head())
print(file_features.shape)

# create andata 

andata = sc.AnnData(
    X=file_matrix.T.tocsr()
)

# CELLS NAMES

andata.obs_names = file_barcodes.iloc[:, 0].astype(str).values

# GENES NAMES 

andata.var_names = file_features.iloc[:, 1].astype(str).values


print("NUMBER OF CELLS:", andata.n_obs)
print("NUMBER OF GENES:", andata.n_vars)


print(andata.X.shape)

andata.var_names_make_unique()

print(andata.var_names.is_unique)

# MITOCHODNRILA GENES

andata.var_names = file_features.iloc[:, 1].astype(str).values
andata.var_names_make_unique()

andata.var["mt"] = andata.var_names.str.upper().str.startswith("MT-")

print("NUMBER OF MITOCHONDRIAL GENES:", andata.var["mt"].sum())


# COUNT MATRIX 

sc.pp.calculate_qc_metrics(
    andata,
    qc_vars=["mt"],
    inplace=True
)


print(andata.obs[
    ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
]
)

# VIOLOIN PLOT 

sc.pl.violin(
    andata,
    ["total_counts","n_genes_by_counts","pct_counts_mt"],
    jitter = 0.4,
    multi_panel = True,
)

# FILTER GENES 

andata = andata[andata.obs["n_genes_by_counts"]>200].copy()

sc.pp.filter_genes(andata,min_cells=3)

print("NUMBER OF CELLS AFTER QC:",  andata.n_obs)
print("NUMBER OF GENES AFTER QC:",  andata.n_vars)

andata.layers["counts"] = andata.X.copy()

# NORMALIZATION 

sc.pp.normalize_total(andata, target_sum=1e4)

# LOG TRANSFORMATION

sc.pp.log1p(andata)

andata.raw = andata.copy()

print("Log-normalized data saved in .raw")

# HIGHLY VARIBALE GENES 

sc.pp.highly_variable_genes(
    andata,
    n_top_genes=3000,
    flavor ="seurat"
)

print("NUMBER OF HVGs:")

andata = andata[
    :,
    andata.var["highly_variable"]
].copy()

# SCALING 

sc.pp.scale(andata, max_value=10)

# PCA 

sc.pp.pca(andata, n_comps=50, svd_solver="arpack")

sc.pl.pca(andata)

# NEIGHBOURS 

# sc.pl.pca_variance_ratio(andata,log=True)

sc.pp.neighbors(andata,n_neighbors=10, n_pcs=30)

# UMAP

sc.tl.umap(andata)

# LEIDEN 

sc.tl.leiden(andata, resolution=0.5)

# sc.pl.umap(andata, color =["leiden"])

# MARKER GENES 

sc.tl.rank_genes_groups(
    andata,
    groupby="leiden",
    method="wilcoxon",
     use_raw=True
)

# sc.pl.rank_genes_groups(
#     andata,
#     n_genes=10,
#     sharey=False
# )


markers = sc.get.rank_genes_groups_df(
         andata,
         group=None
)

# print(markers)


cluster_names = {
    "0": "Fibroblasts",
    "1": "Plasma cells (IgL+)",
    "2": "Low-quality cells",
    "3": "T cells",
    "4": "Goblet cells",
    "5": "Colonocytes",
    "6": "IgG plasma cells",
    "7": "Malignant epithelial",
    "8": "Proliferating tumor cells",
    "9": "B cells",
    "10": "Macrophages/Monocytes",
    "11": "Dendritic cells",
    "12": "Mast cells",
}

# CREATE NEW ANNOTATION COLUMNS 

andata.obs["cell_type"] = (
    andata.obs["leiden"]
    .map(cluster_names)
    .astype("category")
)

print("CELL TYPE COUNTS:")
print(andata.obs["cell_type"].value_counts())



# cell-Type Annotation 

sc.pl.umap(
    andata,
    color="cell_type",
    legend_loc = "on data",
    frameon=False,
    title="Cell Type Annotation"
)

# UMAP with Leiden + cell type side by side

sc.pl.umap(
    andata,
    color=["cell_type"],
    frameon=False,
    wspace=0.4
)


# FIBROBLAST CELLS HAVE THE HIGHEST NUMBER FOLLOWED BY Plasma cells (IgL+) 

Fibro = andata[andata.obs["cell_type"] == "Fibroblasts"].copy()

print(Fibro)
print("Number of fibroblasts:", Fibro.n_obs)
print("Number of genes:", Fibro.n_vars)



# SUB-CLUSTERING OF FIBROBLASTS 

sc.tl.rank_genes_groups(
    Fibro,
    groupby="leiden",
    method="wilcoxon",
    use_raw = True
)

fibro_markers = sc.get.rank_genes_groups_df(
    Fibro,
    group=None
)

print(fibro_markers.head(20))








