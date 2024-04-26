import pandas as pd
import numpy as np
import decoupler as dc
import scanpy as sc
from scipy.stats import norm


from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import liana as li

def encode_y(y):
    le = LabelEncoder()
    le.fit(y)
    return le.transform(y)

def get_ground_truth():
    gt = pd.read_csv("../kuppe/results/edges_Myeloid.csv")[['source', 'target']]
    gt = li.rs.explode_complexes(gt, SOURCE='source', TARGET='target')
    gt = np.union1d(gt['source'], gt['target'])
    
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
                              groups_col=groupby,
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