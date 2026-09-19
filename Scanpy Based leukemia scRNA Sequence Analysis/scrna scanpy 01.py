
import matplotlib.pylab as plt
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns


file_path = r"C:\Users\Shyamendra Sinha\Downloads\GSE329598_Leukemic_cell_lines_scRNAseq_filtered_feature_bc_matrix.h5"

andata =sc.read_10x_h5(file_path)

andata.var_names_make_unique()

print(andata.var_names.is_unique)

# PRINT NUMBER OF CELLS AND GENES

print("NUMBER OF CELLS :", andata.n_obs)
print("NUMBER OF GENES:", andata.n_vars)

#print(andata.obs.head())
#print(andata.var.head())

print(andata.obs_names[:5])

# IDENTIFY MITOCHONDRIAL GENES 

mt_genes = andata.var_names.str.startswith("MT-")

print("Number of mitochondrial genes:", mt_genes.sum())

andata.var["mt"] = mt_genes

# QUALITY CONTROL 

sc.pp.calculate_qc_metrics(andata, qc_vars=["mt"], inplace=True)

print(andata.obs[
        ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
    ]
)

# visulaize QC

sc.pl.violin(
    andata,
    ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
    jitter = 0.4,
    multi_panel=True
)

# REMOVE LOW QUALITY CELLS

andata = andata[andata.obs["n_genes_by_counts"] > 200].copy()

andata = andata[andata.obs["pct_counts_mt"] < 20].copy()

sc.pp.filter_genes(andata,min_cells=3)

# AFTER QUALITY CONTROL 

print("NUMBER OF CELL AFTER QC:",andata.n_obs)
print("NUMBER OF GENES AFTER QC:",andata.n_vars)

# SAVE RAW COUNTS

andata.layers["counts"] = andata.X.copy()

# NORMALIZATION 

sc.pp.normalize_total(andata,target_sum = 1e4)

sc.pp.log1p(andata)

# HIGHLY VARIABLE GENES 

sc.pp.highly_variable_genes(andata,n_top_genes=3000,flavor="seurat")

#sc.pl.highly_variable_genes(andata)

# SCALING 

sc.pp.scale(andata,max_value =10)

# PRINICIPAL COMPONENT ANALYSIS 

sc.pp.pca(andata, n_comps=50, svd_solver="arpack")

sc.pl.pca(andata)

# PCA VARINACE PLOT 

#sc.pl.pca_variance_ratio(andata, log=True)

# NEAREST NEIGHBOUR 

sc.pp.neighbors(andata,n_neighbors =10, n_pcs=30)

# UMAP

sc.tl.umap(andata)

sc.pl.umap(andata)

# LEIDEN CLSUTERING AND WILCOXON 

sc.tl.leiden(andata, resolution=0.5)

#sc.pl.umap(
#   andata,
#    color=["leiden"]
#)


# MARKER GENES 

sc.tl.rank_genes_groups(
    andata, 
    groupby="leiden",
    method="wilcoxon"
)

sc.pl.rank_genes_groups(
    andata,
    n_genes=10,
    sharey= False
)

marker = sc.get.rank_genes_groups_df(
    andata,
    group=None
)

marker.to_csv(
    "cluster_marker_genes.csv",
    index=False
)

# TAKING CLUSTER_3 REPRESENTING T-CELL 

markers_3 = sc.get.rank_genes_groups_df(
    andata,
    group ="3"
)

top_genes_3 = markers_3["names"].head(10).tolist()

print(top_genes_3)


sc.pl.heatmap(
    andata,
    var_names=top_genes_3,
    groupby="leiden",
    standard_scale="var"
)

# SAVE PROCESSED DATA 

andata.write("single_cell_processed.h5ad")

print("Final number of cells:", andata.n_obs)
print("Final number of genes:", andata.n_vars)

print("Cluster information:")

print(andata.obs["leiden"].value_counts())