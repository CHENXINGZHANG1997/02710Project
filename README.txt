Support Vector Machines with Gaussian Elastic Metric Kernels for Time Series Classification of gene expression data. Pyhthon Implementation

This implementation can be used with two types of distance functions:
TWED: Time Warp Edit Distance
ERP: Edit distance with Real Penalty

Requires PyML and numPy installed

USAGE: Requires dataset in excel format. See Dataset_s1.xls for example.
kernel can be either "gerp" or "twed"

Command: python executeAnalysis dataset kernel

The result will be in the respective analysis file (analysisGERP.txt or analysisTWED.txt) after model selection. Model Parameters will be given along with accuracy for classification data uptil each time point.
