
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

pdf (
  file = "AML_SUMMARY.pdf",
  width = 10,
  height = 12
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

pdf(
  file = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/ONCOPLOT.pdf",
  width = 12,
  height = 10
)
oncoplot(
  maf = AML_maf,
  top = 30
)

dev.off()
AML_maf@data$Hugo_Symbol


# TRANSVERISON / TRANSITION MUTATION 

pdf(
  file = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/TITV PLOT.pdf",
  width = 10,
  height = 12
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

pdf(
  file = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/SOMATIC_INTERACTION.pdf",
  width = 10,
  height = 12
)

somaticInteractions(
  maf = AML_maf,
  top = 30,
  pvalue = c(0.05, 1)
)


dev.off()



# count each mutation 

table(MAF_data$Variant_Classification)

# LOLLIPOPLOT 

dev.new()

pdf(
  file = "C:/Users/Shyamendra Sinha/Desktop/R programming files/R FILES/AML MAFTOOLS ANALYSIS/LOLLIPOPLOT_NMP1_GENE.pdf",
  width = 8,
  height = 9
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


# VARIANT ALLELE FREQUENCY 

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




































