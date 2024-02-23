import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

import scanpy as sc
import liana as li

from itertools import product

from memory_profiler import memory_usage
from timeit import default_timer as timer


def _sample_anndata(sparsity = 0.90, n_ct = 10, n_vars = 2000, n_obs = 1000, seed=1337):    
    rng = np.random.default_rng(seed=seed)
    counts = rng.poisson(100, size=(n_obs, n_vars))
    mask = rng.choice([0, 1], size=(n_obs, n_vars), p=[sparsity, 1 - sparsity])
    counts = csr_matrix(counts * mask, dtype=np.float32)
    
    adata = ad.AnnData(counts)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    adata.var_names = [f"Gene{i:d}" for i in range(adata.n_vars)]
    adata.obs_names = [f"Cell{i:d}" for i in range(adata.n_obs)]
    print(f" NNZ fraction: {adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])}")
    
    x = rng.integers(low=0, high=5000, size=adata.shape[0])
    y = rng.integers(low=0, high=5000, size=adata.shape[0])
    adata.obsm['spatial'] = np.array([x, y]).T
    
    li.ut.spatial_neighbors(adata, cutoff=0.1, bandwidth=150, max_neighbours=10)
    
    # assign cell types
    ct = rng.choice([f"CT{i:d}" for i in range(n_ct)], size=(adata.n_obs,))
    ct = rng.choice(ct, size=(adata.n_obs,))
    adata.obs["cell_type"] = pd.Categorical(ct)
    
    return adata


def _sample_resource(adata, n_lrs = 3000, seed=1337):
    resource = pd.DataFrame(product(adata.var_names, adata.var_names)).rename(columns={0: "ligand", 1: "receptor"})
    resource = resource[resource["ligand"] != resource["receptor"]]
    resource = resource.sample(n_lrs, replace=False, random_state=seed)
    return resource

def _benchmark(function, **kwargs):
    # Define a wrapper function that will be passed to memory_usage
    def wrapper():
        function(**kwargs)

    start = timer()
    peak_memory = memory_usage(proc=wrapper, max_usage=True, include_children=True)

    end = timer()
    time_taken = end - start

    return time_taken, peak_memory
