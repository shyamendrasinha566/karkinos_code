library(maftools)
library(TCGAbiolinks)

# SEARCH FOR THE DATA 

query <- GDCquery(
  project = "TCGA-LAML",
  data.category = "Simple Nucleotide Variation",
  data.type = "Masked Somatic Mutation",
  workflow.type = "Aliquot Ensemble Somatic Variant Merging and Masking",
  access = "open"
)

# CHECK AVAILABLE FILES 

getResults(query)
dim(query)
head(query)

# DOWNLOAD GDC FILE 

GDCdownload(
  query = query,
  method = "api",
  directory = "GDCdata"
)

# PREPARE GDC DATA 

MAF_data <- GDCprepare(
  query = query,
  directory = "GDCdata"
) 

class(MAF_data)
dim(MAF_data)


# IMPORT INTO MAF TOOLS 

AML_maf <- read.maf(MAF_data)
AML_maf

# MUTATION SUMMARY PLOT - VARIANT CLASSIFICATION AND VAIRANT TYPE 

graphics.off()

png(
  filename = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/AML_SUMMARY.png",
  width = 2500,
  height = 3000,
  res = 300
)

plotmafSummary(
  maf = AML_maf,
  rmOutlier = TRUE,
  addStat = "median",
  dashboard = TRUE
)

dev.off()

# ONCOPLOT 

dev.new()
png(
  filename = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/AML_ONCOPLOT.png",
  width = 2500,
  height = 3000,
  res = 300
)
oncoplot(
  maf = AML_maf,
  top = 30
)

dev.off()

# TRANSVERISON / TRANSITION MUTATION 

png(
  filename = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/AML_TITV_PLOT.png",
  width = 2500,
  height = 3000,
  res = 300
)

# Generate Ti/TV plot

TITV <- titv(
  maf = AML_maf,
  plot = TRUE,
  useSyn = TRUE
)

dev.off()

# TUMOR MUTATIONAL BURDERN (TMB)

TMB_results <- tmb(
  maf= AML_maf,
  captureSize = 50
)

summary(TMB_results)

head(TMB_results)


# SOMATIC INTERACTIONS 

png(
  filename = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/AML_SOMATIC_INTERACTION.png",
  width = 2500,
  height = 3000,
  res = 300
)
somaticInteractions(
  maf = AML_maf,
  top = 30,
  pvalue = c(0.05, 1)
)


dev.off()

# count each mutation 

table(MAF_data$Variant_Classification)

# GENE WISE MUTATION ANALYSIS 

gene_summary <- getGeneSummary(AML_maf)

gene_summary

top_genes <- gene_summary[1:20, ]

# LOLLIPOPLOT 

dev.new()

png(
  filename = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/AML_NPM1_LOLLIPOPLOT.png",
  width = 2500,
  height = 3000,
  res = 300
)

lollipopPlot(
  maf = AML_maf,
  gene = "NPM1",
  AACol = "HGVSp_Short",
  labelPos = "all",
)

title(
  main = "NPM1 Mutation Profile in AML",
  cex.main = 1.2,
  font.main = 2
)


dev.off()


# VARIANT ALLELE FREQUENCY (VAF)

AML_vaf <- MAF_data[
  !is.na(MAF_data$t_alt_count)&
  !is.na(MAF_data$t_ref_count)&
    (MAF_data$t_alt_count + MAF_data$t_ref_count) > 0,
]

AML_vaf$vaf <- with(
  MAF_data,
  t_alt_count / (t_alt_count + t_ref_count)
)

# INTO PERCENTAGE VAF

AML_vaf$vaf_percent <- AML_vaf$vaf * 100

summary(AML_vaf$vaf)

dev.new()

pdf (
  "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/HISTORGRAM_VAF_PLOT.pdf",
   width = 8,
   height = 9
)

hist(
  AML_vaf$vaf,
  breaks = 50,
  main = "VAF Distribution in AML",
  xlab = "Variant Allele Frequency",
  col = "skyblue",
  border = "black"
)

dev.off()


# MATH SCORE - INTRATUMORAL HETEROGENICITY 

math_results <- math.score(
  maf = AML_maf
)

head(math_results)

# ONCODRIVE
# TO IDENTIFY GENES WITH SIGNIFICNAT CLUSTERING OF MUTATIONS ALONG THE GENOME POSITIONS

oncodrive_results <- oncodrive(
  maf = AML_maf,
  AACol = "HGVSp_Short",
  minMut = 5,
  pvalMethod = "zscore"
)

colnames(oncodrive_results)
summary (oncodrive_results)


# ONCOGENIC PATWAY ANALYSIS 

png(
  filename = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/AML_PATHWAYS_PLOT.png",
  width = 2500,
  height = 3000,
  res = 300
)

AML_pathways <- pathways(
  maf = AML_maf,
  plotType = "bar"
)

plotPathways(
  maf = AML_maf,
  pathlist = AML_pathways
)

dev.off()

# DRUG-GENE INTERACTIONS 

png(
  filename = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/AML_DGI_PLOT.png",
  width = 2500,
  height = 3000,
  res = 300
)

DGI_RESULTS <- drugInteractions(
  maf = AML_maf,
  top = 25,
  plotType = "bar",
  fontSize = 1
)

dev.off()

