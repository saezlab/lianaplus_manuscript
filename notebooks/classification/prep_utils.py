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


def map_gene_symbols(adata, map_df):    
    """Maps gene symbols from aliases to standard symbols
    Genes that map many-to-one are summed.
    Genes that map one-to-many are duplicated.
    
    Adapted from Scott Gigante:
    https://github.com/openproblems-bio/openproblems/blob/e7513007c145bcb9c1c1ae81b6e00c41051ad11d/openproblems/tasks/_cell_cell_communication/_common/utils.py#L34
    
    Parameters
    ----------
    adata : anndata.AnnData
    map_df :
        dataframe containing gene symbol map with two columns, `gene` and `alias`
    Returns
    -------
    adata : anndata.AnnData
    """
    
    import collections
    import anndata
    import scipy
    
    var = adata.var.rename_axis("alias", axis=0)[[]]
    gene_match_idx = np.isin(var.index, map_df["gene"]) # NOTE: this is very slow
    
    
    var_gene_match, var = var.loc[gene_match_idx].copy(), var.loc[~gene_match_idx]
    alias_match_idx = np.isin(var.index, map_df["alias"])
    var_alias_match, var_no_map = (
        var.loc[alias_match_idx].copy(),
        var.loc[~alias_match_idx].copy(),
    )

    # fill 'gene' column
    var_alias_match = var_alias_match.reset_index().merge(
        map_df, on="alias", how="left"
    )
    var_gene_match["gene"] = var_gene_match.index
    var_no_map["gene"] = var_no_map.index

    var_dealiased = pd.concat(
        [var_gene_match.reset_index(), var_no_map.reset_index(), var_alias_match]
    )
    duplicate_idx = var_dealiased["gene"].duplicated(keep=False)
    var_dealiased_many_to_one, var_dealiased_one_to_any = (
        var_dealiased.loc[duplicate_idx],
        var_dealiased.loc[~duplicate_idx],
    )

    adata_one_to_any = adata[:, var_dealiased_one_to_any["alias"]]
    adata_one_to_any.var.index = var_dealiased_one_to_any["gene"]

    many_to_one_genes = var_dealiased_many_to_one["gene"].unique()
    many_to_one_X = []
    many_to_one_layers = collections.defaultdict(list)
    for gene in var_dealiased_many_to_one["gene"].unique():
        gene_aliases = var_dealiased_many_to_one.loc[
            var_dealiased_many_to_one["gene"] == gene, "alias"
        ]
        adata_gene = adata[:, gene_aliases]
        many_to_one_X.append(scipy.sparse.coo_matrix(adata_gene.X.sum(axis=1)))
        for layer_name, layer in adata_gene.layers.items():
            many_to_one_layers[layer_name].append(
                scipy.sparse.coo_matrix(adata_gene.X.sum(axis=1))
            )

    return anndata.AnnData(
        X=scipy.sparse.hstack([adata_one_to_any.X] + many_to_one_X).tocsr(),
        obs=adata.obs,
        var=pd.DataFrame(
            index=np.concatenate([adata_one_to_any.var.index, many_to_one_genes])
        ),
        layers={
            layer_name: scipy.sparse.hstack(
                [adata_one_to_any.layers[layer_name]] + many_to_one_layers[layer_name]
            ).tocsr()
            for layer_name in adata.layers
        },
        uns=adata.uns,
        obsm=adata.obsm,
    )