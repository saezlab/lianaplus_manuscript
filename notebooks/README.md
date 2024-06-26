# LIANA+ Analysis Notebooks

This folder contains the notebooks used to generate the figures in the manuscript.

## Notebooks

Within the `classification/` folder, are the notebooks to reproduce the Label Classification results with MOFA+ and Tensor-cell2cell; these make use of datasets (1-5).

Within the `kuppe` folder, are the notebooks to reproduce the results on Myocardial Infarction (1) data

Within the `spot_calling` folder, are the notebooks to reproduce evaluation results of local metrics; classifying malignant spots from breast cancer data (12) and predicting cell type proportions (5).

Within the `sma` folder, are the notebooks to reproduce the results on Spatial-Metabolome Analysis using the murine Parkinson's disease data (7).

Within `efficiecny` folder, are the notebooks to reproduce the results on the efficiency of LIANA+.

Within `slideseq_benchmark` folder, are the notebooks to reproduce the results on the Slide-seq benchmarking (8).


## Datasets

1. [Kuppe et al. (2022)](https://www.nature.com/articles/s41586-022-05060-x) Spatial and single-cell infarction data, obtained via https://cellxgene.cziscience.com/collections/8191c283-0816-424b-9b61-c3e1d6258a77 
2. [Reichart et al. (2022)](https://www.science.org/doi/10.1126/science.abo1984) cardiomyopathies, obtained via https://cellxgene.cziscience.com/collections/e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb 
3. [Carraro et al., 2021](https://www.nature.com/articles/s41591-021-01332-7)  Cystic fibrosis data, obtained via https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150674 
4. [Habermann et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32832598/) Pulmanory Fibrosis data, obtained via https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135893 
5. [Velmeshev et al., 2019](https://www.science.org/doi/full/10.1126/science.aav8130) Single-cell Autism Spectrum Disorder data, obtained via https://codeocean.com/capsule/9737314/tree/v2; https://www.ncbi.nlm.nih.gov/bioproject/PRJNA434002/ 
6. [Wu et al., 2021](https://www.nature.com/articles/s41588-021-00911-1) Processed breast cancer 10× Visium slides are available at https://zenodo.org/record/4739739. 
7. [Vicari et al, 2023](https://www.nature.com/articles/s41587-023-01937-y) Parkinson's disease data, obtained via https://data.mendeley.com/datasets/w7nw4km7xd/1
8. [Russell et al., 2023](https://www.nature.com/articles/s41586-023-06837-4) Slide-Seq datasets were obtained via the Broad Institute Single Cell Portal: mouse embryonic brain - [SCP2170](https://singlecell.broadinstitute.org/single_cell/study/SCP2170); mouse brain - [SCP2162](https://singlecell.broadinstitute.org/single_cell/study/SCP2162)  human brain: [SCP2167](https://singlecell.broadinstitute.org/single_cell/study/SCP2167), human tonsil [SCP2169](https://singlecell.broadinstitute.org/single_cell/study/SCP2169); [SCP2171](https://singlecell.broadinstitute.org/single_cell/study/SCP2171); and human melanoma multiome - [SCP2176](https://singlecell.broadinstitute.org/single_cell/study/SCP2176); also available under GEO GSE244355.
9. Datasets used for the reliability of predictions:

| Dataset   | Samples | Condition                         | Reference | Data URL                                                                                                      |
|-----------|---------|-----------------------------------|-----------|---------------------------------------------------------------------------------------------------------------|
| Kuppe     | 23      | Acute cardiac Infarction          | https://www.nature.com/articles/s41586-022-05060-x        | [Link](https://cellxgene.cziscience.com/collections/8191c283-0816-424b-9b61-c3e1d6258a77)                    |
| Reichart  | 126     | Cardiomyopathies                  | https://www.science.org/doi/10.1126/science.abo1984        | [Link](https://cellxgene.cziscience.com/collections/e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb)                    |
| Simonson  | 15      | Ischemic cardiomyopathy           | https://www.sciencedirect.com/science/article/pii/S2211124723000979?via%3Dihub        | [Link](https://singlecell.broadinstitute.org/single_cell/study/SCP1849/)                                      |
| Koenig    | 38      | dilated (nonischemic) cardiomyopathy | https://www.nature.com/articles/s44161-022-00028-6      | [Link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183852)                                          |
| Chaffin   | 42      | dilated and hypertrophic cardiomyopathy | https://www.nature.com/articles/s41586-022-04817-8    | [Link](https://singlecell.broadinstitute.org/single_cell/study/SCP1303/)                                      |
| Armute    | 40      | Cardiomyopathies                  | https://www.nature.com/articles/s44161-023-00260-8        | [Link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226314)                                          |


‡ Number of samples included in the evaluation				