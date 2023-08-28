# LIANA+ Analysis Notebooks

This folder contains the notebooks used to generate the figures in the manuscript.

## Notebooks

Within the `applications/` folder, you will find the following notebooks:
- `liana_c2c.ipynb`: An application of Tensor-cell2cell to single-cell lupus patient PBMCs data (1).
- `mofatalk.ipynb`: An application of MOFA and LIANA+ to single-cell lupus patient PBMCs data (1).
- `targeted`: An application of PyDESeq2 and inference of putatively causal networks in single-cell lupus patient PBMCs (1).
- `sc_citeseq.ipynb`: An application to healthy COVID-19 single-cell CITE-seq PBMCs data (2).
- `spatial_citeseq.ipynb`: An application to tonsil CITE-seq spatial data (3).
- `spatial_metalinks.ipynb`: An application of metabolite-mediated CCC inference to mouse brain spatial data (4).

Within the `classification/` folder, are the notebooks to reproduce the Label Classification results with MOFA+ and Tensor-cell2cell; these make use of datasets 5-9.

Within the `scAKI` and `visiumAKI` folders, are the notebooks to reproduce the acute murine injury results on Single-cell (10) and Spatially-resolved (11) data.

Within the `kuppe_visium` folder, are the notebooks to reproduce the results on Myocardial Infarction (5) data

Within the `spot_calling` folder, are the notebooks to reproduce evaluation results of local metrics; classifying malignant spots from breast cancer data (12) and predicting cell type proportions (5).

## Datasets

1. [Kang et al. (2019)](https://www.nature.com/articles/nbt.4042) Lupus PBMCs data was obtained as a processed AnnData object from https://figshare.com/ndownloader/files/34464122; available via [pertpy](https://github.com/theislab/pertpy).
2. [Chang Zuckerberg Consortia (2020)](https://www.medrxiv.org/content/10.1101/2020.11.20.20227355v1), obtained available via https://covid19.cog.sanger.ac.uk/submissions/release2/vento_pbmc_processed.h5ad
3. [Liu et al. (2023)](https://www.nature.com/articles/s41587-023-01676-0), Spatially-resolved tonsil CITE-seq data 42 (GSE213264) was obtained via https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213264. 
4. A 10X Visium slide (available at https://support.10xgenomics.com/spatial-gene-expression/datasets) was obtained via [Squidpy](https://github.com/scverse/squidpy).
5. [Kuppe et al. (2022)](https://www.nature.com/articles/s41586-022-05060-x) Spatial and single-cell infarction data, obtained via https://cellxgene.cziscience.com/collections/8191c283-0816-424b-9b61-c3e1d6258a77 
6. [Reichart et al. (2022)](https://www.science.org/doi/10.1126/science.abo1984) cardiomyopathies, obtained via https://cellxgene.cziscience.com/collections/e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb 
7. [Carraro et al., 2021](https://www.nature.com/articles/s41591-021-01332-7)  Cystic fibrosis data, obtained via https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150674 
8. [Habermann et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32832598/) Pulmanory Fibrosis data, obtained via https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135893 
9. [Velmeshev et al., 2019](https://www.science.org/doi/full/10.1126/science.aav8130) Single-cell Autism Spectrum Disorder data, obtained via https://codeocean.com/capsule/9737314/tree/v2; https://www.ncbi.nlm.nih.gov/bioproject/PRJNA434002/ 
10. [Kirita et al., 2020](https://www.pnas.org/doi/epdf/10.1073/pnas.2005477117) Single-nuc mouse kidney injury, obtained via GSE139107 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139107)
11. [Dixon et al. 2022](https://journals.lww.com/jasn/pages/articleviewer.aspx?year=2022&issue=02000&article=00005&type=Fulltext) Spatially-resolved mouse kidney injury data; obtained via GSE182939 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182939).
12. [Wu et al., 2021](https://www.nature.com/articles/s41588-021-00911-1) Processed breast cancer 10× Visium slides are available at https://zenodo.org/record/4739739. 