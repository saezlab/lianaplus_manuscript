import os
import pandas as pd
import numpy as np
import decoupler as dc
import scanpy as sc
from scipy.stats import norm, zscore

from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import liana as li

def load_prep_slide(path, slide, add_sample_name=False, min_genes = 400, bandwidth=150, cutoff=0.1, set_diag=True, **kwargs):
    adata = sc.read_h5ad(os.path.join(path, slide))
    
    if add_sample_name:
        indeces = [f"{slide.split('.')[0]}-{i}" for i in adata.obs.index]
        adata.obs.index = indeces
        adata.obsm['compositions'].index = indeces
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=5)
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    li.ut.spatial_neighbors(adata, bandwidth=bandwidth, cutoff=cutoff, set_diag=set_diag, **kwargs)
    
    return adata


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


def encode_y(y):
    le = LabelEncoder()
    le.fit(y)
    return le.transform(y)

def get_ground_truth(df, source='ligand_complex', target='receptor_complex'):
    gt = df[[source, target]]
    gt = li.rs.explode_complexes(gt, SOURCE=source, TARGET=target)
    gt = np.union1d(gt[source], gt[target])
    
    return gt
    
    
def process(adata, 
            sample_key='sample_id',
            groupby='cell_type',
            condition_key='heart_failure',
            myeloid_label='Myeloid',
            layer=None):
    
    adata = adata[adata.obs[groupby]==myeloid_label]
    pdata = dc.get_pseudobulk(adata,
                              sample_col=sample_key,
                              groups_col=None,
                              min_cells=10,
                              min_counts=1000,
                              layer=layer,
                              )

    # Store raw counts in layers
    pdata.layers['counts'] = pdata.X.copy()

    # Normalize and scale
    sc.pp.normalize_total(pdata, target_sum=1e4)
    sc.pp.log1p(pdata)
    sc.pp.scale(pdata, max_value=10)
    
    y_true = encode_y(pdata.obs[condition_key])
    
    return pdata, y_true


def process_kuppe():
    groupby = 'cell_type_original'
    sample_key = 'sample'
    condition_key = 'patient_group'
    myeloid_label = 'MY'
    layer = 'counts'
    
    kuppe = sc.read_h5ad('../kuppe/results/kuppe_processed.h5ad')
    kuppe = kuppe[kuppe.obs[condition_key].isin(['myogenic', 'ischemic'])]
    
    return process(adata=kuppe,
                    sample_key=sample_key,
                    myeloid_label=myeloid_label,
                    condition_key=condition_key,
                    groupby=groupby,
                    layer=layer)

def process_reichart():
    reichart = sc.read_h5ad('../classification/data/interim/reichart_processed.h5ad')
    groupby='cell_type'
    sample_key='Sample'
    condition_key='disease'
    myeloid_label = 'myeloid cell'
    layer = 'counts'
    
    return process(adata=reichart,
                    sample_key=sample_key,
                    myeloid_label=myeloid_label,
                    condition_key=condition_key,
                    groupby=groupby,
                    layer=layer
                    )

def process_simonson():
    simonson = sc.read_h5ad("../../data/Simonson2023_ICM.h5ad")
    return process(simonson)

def process_koenig():
    simonson = sc.read_h5ad("../../data/Koenig2022_DCM.h5ad")
    return process(simonson)

def process_armute():
    armute = sc.read_h5ad("../../data/Armute2023_LVAD.h5ad")
    return process(armute)

def process_chaffin():
    armute = sc.read_h5ad("../../data/Chaffin2022_DCM.h5ad")
    return process(armute)

def calculate_p_value(gt, random):
    # make a distribution out of random
    mu, std = norm.fit(random)
    # Calculate the z-score of the original value
    z_score = (gt - mu) / std
    # Calculate the probability of getting a value >= original_value under the fitted distribution
    p_value = 1 - norm.cdf(gt, mu, std)
    
    return p_value, z_score