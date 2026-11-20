# Spatial and epigenetic programs underlie heterogeneous cellular senescence in the aging liver

This repository organizes the analysis code for the following study of cellular senescence in the aging mouse liver using 10x Multiome, Paired-Tag, snm3C-seq, and MERFISH.

Mamde\*, Li\*, Teneche\*, et al. “Spatial and Epigenetic Programs Underlie Heterogeneous Cellular Senescence in the Aging Liver.” 19 Nov. 2026. *bioRxiv*. [doi:10.64898/2026.11.18.767706](https://doi.org/10.64898/2026.11.18.767706).



![Figure 1A–B. Aging mouse liver study design and the four profiling assays.](img/Figure1AB.png)

## Analysis workflows

| Directory | Analysis |
| --- | --- |
| [`analysis/SenLearner/`](analysis/SenLearner/) | Identify senescent cells from 10x Multiome RNA profiles. |
| [`analysis/MERFISH/processing/`](analysis/MERFISH/processing/) | Prepare MERFISH expression, cell annotations, and spatial coordinates. |
| [`analysis/MERFISH/SnC_mapping/`](analysis/MERFISH/SnC_mapping/) | Use the 10x Multiome reference and SenLearner outputs to map senescent cells into MERFISH. |
