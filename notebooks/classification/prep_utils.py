from scipy.stats import zscore

def filter_samples(adata, sample_key, condition_key, 
                   min_cells_per_sample, sample_zcounts_max,
                   sample_zcounts_min):
    """
    Filter out samples with too few cells or too many cells.
    
    Parameters
    ----------
    adata: AnnData
        AnnData object with samples in obs[sample_key]
    
    
    """
    
    
    # calculate total sample counts & zscores
    sample_tcounts = adata.obs[sample_key].value_counts()
    sample_zcounts = zscore(sample_tcounts)
    
    # get samples that pass the threshold
    samples_msk = (sample_tcounts > min_cells_per_sample) & \
        (sample_zcounts < sample_zcounts_max) & \
            (sample_zcounts > sample_zcounts_min)
    samples_keep = samples_msk[samples_msk].index

    ## Filter out samples with too few cells
    adata = adata[adata.obs[sample_key].isin(samples_keep), :]
    print(adata.obs[[sample_key, condition_key]].drop_duplicates().groupby(condition_key).count())

    return adata


def filter_celltypes(adata, groupby, sample_key, min_cells, min_samples):
    # get cell types that pass the threshold
    # get cell num per sample per cluster
    celltype_qc = adata.obs.groupby([sample_key, groupby]).size().reset_index(name='counts')
    # check which rows (sample-cell type combo) pass threshold
    celltype_qc['keep_min'] = celltype_qc['counts'] >= min_cells
    # how many samples passed the threshold
    celltype_qc['keep_sum'] = celltype_qc.groupby(groupby)['keep_min'].transform('sum')
    # identify which cell types don't pass sample threshold
    celltype_qc['keep_celltype'] = celltype_qc['keep_sum'] >= min_samples
        
        
    celltypes_keep = celltype_qc[celltype_qc['keep_celltype']][groupby].unique()

    # filter out cell types with too few cells
    adata = adata[adata.obs[groupby].isin(celltypes_keep), :]
    
    return adata