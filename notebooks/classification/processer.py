import os
import numpy as np
import scanpy as sc
from prep_utils import filter_samples, filter_celltypes, check_group_balance, map_gene_symbols
from scipy.sparse import csr_matrix, issparse
import liana as li

class DatasetHandler:
    def __init__(self, dataset_name):
        self.dataset_params = {
            'defaults': {
                'groupby': None,
                'sample_key': None,
                'condition_key': None,
                'batch_key': None,
                'min_cells_per_sample': 1000,
                'sample_zcounts_max': 3,
                'sample_zcounts_min': -2,
                'min_cells': 20,
                'min_samples': 5,
                'use_raw': None,
                'change_var_to': None,
                'conditions_to_keep': None,
                'organism':'human'
            },
            'kuppe': {
                'groupby': 'cell_type',
                'sample_key': 'sample',
                'condition_key': 'patient_group',
                'batch_key': 'sex',
                'use_raw': True,
                'change_var_to': 'feature_name',
                'conditions_to_keep': ['ischemic', 'myogenic']
            },
            'habermann': {
                'groupby': 'cell_type',
                'sample_key': 'sample',
                'condition_key': 'patient_group',
                'batch_key': 'sex',
            },
            'reichart': {
                'groupby':'celltype',
                'sample_key':'Sample',
                'condition_key':'disease',
                'batch_key':'Sample_Source',
                'conditions_to_keep':['normal', 'dilated cardiomyopathy']
            },
            'velmeshev':{
                'groupby':'cluster',
                'sample_key':'individual',
                'condition_key':'diagnosis',
                'batch_key':'sex'
            },
            'carraro': {
                'groupby': 'major',
                'sample_key': 'orig.ident',
                'condition_key': 'type',
                'batch_key': 'lab',
            }
        }
        
        if dataset_name not in self.dataset_params:
            raise ValueError(f"Invalid dataset name '{dataset_name}'. Please provide a valid dataset name.")
        
        # ensure that groupby, sample_key, and condition_key are not None
        if self.dataset_params[dataset_name]['groupby'] is None:
            raise ValueError(f"Please provide a valid 'groupby' parameter.")
        if self.dataset_params[dataset_name]['sample_key'] is None:
            raise ValueError("Please provide a valid 'sample_key' parameter.")
        if self.dataset_params[dataset_name]['condition_key'] is None:
            raise ValueError("Please provide a valid 'condition_key' parameter.")
        
        dataset_info = self.dataset_params[dataset_name]
        
        defaults = self.dataset_params['defaults']
        
        self.dataset_name = dataset_name
        self.groupby = dataset_info.get('groupby', defaults['groupby'])
        self.sample_key = dataset_info.get('sample_key', defaults['sample_key'])
        self.condition_key = dataset_info.get('condition_key', defaults['condition_key'])
        self.batch_key = dataset_info.get('batch_key', defaults['batch_key'])
        self.min_cells_per_sample = dataset_info.get('min_cells_per_sample', defaults['min_cells_per_sample'])
        self.sample_zcounts_max = dataset_info.get('sample_zcounts_max', defaults['sample_zcounts_max'])
        self.sample_zcounts_min = dataset_info.get('sample_zcounts_min', defaults['sample_zcounts_min'])
        self.min_cells = dataset_info.get('min_cells', defaults['min_cells'])
        self.min_samples = dataset_info.get('min_samples', defaults['min_samples'])
        self.use_raw = dataset_info.get('use_raw', defaults['use_raw'])
        self.change_var_to = dataset_info.get('change_var_to', defaults['change_var_to'])
        self.conditions_to_keep = dataset_info.get('conditions_to_keep', defaults['conditions_to_keep'])
        
    
    def process_dataset(self):
        
        adata = sc.read_h5ad(os.path.join('data', f"{self.dataset_name}.h5ad"), backed='r')
        
        if self.conditions_to_keep is not None:
            msk = np.array([patient in self.conditions_to_keep for patient in adata.obs[self.condition_key]])
            adata = adata[msk]
        
        if self.use_raw:
            adata = adata.raw.to_adata()
        
        if not issparse(adata.X):
            adata.X = csr_matrix(adata.X)
            
            
        if self.dataset_name=='velmeshev':
            #TODO as param map_var that accepts a csv path
            df = adata.var.reset_index()['index'].str.split('\\|', expand=True).rename(columns={0:'ensembl', 1:'genesymbol'})
            adata.var = df.set_index('ensembl')
            map_df = df.rename(columns={'ensembl':'alias', 'genesymbol':'gene'})
            map_df
            adata = map_gene_symbols(adata, map_df)
            
        # change to gene symbols
        if self.change_var_to is not None:
            adata.var.index = adata.var[self.change_var_to]
            
        adata = filter_samples(adata, 
                               sample_key = self.sample_key,
                               condition_key = self.condition_key,
                               min_cells_per_sample= self.min_cells_per_sample,
                               sample_zcounts_max= self.sample_zcounts_max,
                               sample_zcounts_min= self.sample_zcounts_min)
        
        adata = check_group_balance(adata,
                                    condition_key = self.condition_key,
                                    sample_key = self.sample_key)
        
        adata = filter_celltypes(adata=adata, 
                                 groupby=self.groupby,
                                 sample_key=self.sample_key,
                                 min_cells=self.min_cells,
                                 min_samples=self.min_samples)
        
        # Remove genes expressed in few cells, normalize
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        adata
        
        # Normalize
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Run LIANA (NOTE: should be by method in a sep function)
        li.mt.rank_aggregate.by_sample(adata, 
                                       groupby=self.groupby,
                                       use_raw=False,
                                       sample_key=self.sample_key,
                                       verbose=True,
                                       n_perms=None)
        
        adata.write_h5ad(os.path.join('data', 'interim', f"{self.dataset_name}_processed.h5ad"))
        
        return adata
                