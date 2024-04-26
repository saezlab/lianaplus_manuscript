import os
import numpy as np
import pandas as pd
import scanpy as sc
from pandas import read_csv
from scipy.sparse import csr_matrix, issparse
import liana as li

from liana.method import cellphonedb, connectome, cellchat, scseqcomm, singlecellsignalr, natmi, logfc, rank_aggregate, geometric_mean
methods = [cellphonedb, connectome, cellchat, scseqcomm, singlecellsignalr, natmi, logfc, rank_aggregate, geometric_mean]

from classify_utils import (
    _dict_setup,
    run_mofatalk,
    run_tensor_c2c,
    _run_rf_auc,
    _assign_dict,
    _generate_splits
    )

from prep_utils import (
    filter_samples,
    filter_celltypes,
    check_group_balance,
    map_gene_symbols
    )


class DatasetHandler:
    def __init__(self, dataset_name):
        self.dataset_params = {
            'defaults': {
                'groupby': None,
                'sample_key': None,
                'condition_key': None,
                'min_cells_per_sample': 1000,
                'sample_zcounts_max': 3,
                'sample_zcounts_min': -2,
                'min_cells': 20,
                'min_samples': 5,
                'use_raw': False,
                'change_var_to': None,
                'conditions_to_keep': None,
                'organism':'human',
                'map_path':None,
                "n_factors":10,
            },
            'kuppe': {
                'groupby': 'cell_type_original',
                'sample_key': 'sample',
                'condition_key': 'patient_group',
                'use_raw': True,
                'change_var_to': 'feature_name',
                'conditions_to_keep': ['ischemic', 'myogenic']
            },
            'habermann': {
                'groupby': 'celltype',
                'sample_key': 'Sample_Name',
                'condition_key': 'Status',
            },
            'reichart': {
                'groupby':'cell_type',
                'sample_key':'Sample',
                'condition_key':'disease',
                'conditions_to_keep':['normal', 'dilated cardiomyopathy'],
                "map_path":"ensembl_to_symbol.csv",
                "n_factors":20,
            },
            'velmeshev':{
                'groupby':'cluster',
                'sample_key':'sample',
                'condition_key':'diagnosis',
                # the mapping table was taken from this dataset's .var
                "map_path":"ensembl_to_symbol.csv"
            },
            'carraro': {
                'groupby': 'major',
                'sample_key': 'orig.ident',
                'condition_key': 'type',
                'min_cells_per_sample': 700,
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
        self.min_cells_per_sample = dataset_info.get('min_cells_per_sample', defaults['min_cells_per_sample'])
        self.sample_zcounts_max = dataset_info.get('sample_zcounts_max', defaults['sample_zcounts_max'])
        self.sample_zcounts_min = dataset_info.get('sample_zcounts_min', defaults['sample_zcounts_min'])
        self.min_cells = dataset_info.get('min_cells', defaults['min_cells'])
        self.min_samples = dataset_info.get('min_samples', defaults['min_samples'])
        self.use_raw = dataset_info.get('use_raw', defaults['use_raw'])
        self.change_var_to = dataset_info.get('change_var_to', defaults['change_var_to'])
        self.conditions_to_keep = dataset_info.get('conditions_to_keep', defaults['conditions_to_keep'])
        self.map_path = dataset_info.get('map_path', defaults['map_path'])
        self.n_factors = dataset_info.get('n_factors', defaults['n_factors'])
        
        self.all_datasets = self.dataset_params.keys()
    
    def process_dataset(self):
        
        adata = sc.read_h5ad(os.path.join('data', f"{self.dataset_name}.h5ad"))
        
        if self.conditions_to_keep is not None:
            msk = np.array([patient in self.conditions_to_keep for patient in adata.obs[self.condition_key]])
            adata = adata[msk]
        
        if self.use_raw:
            adata = adata.raw.to_adata()
        
        if not issparse(adata.X):
            adata.X = csr_matrix(adata.X)
            
        # change to gene symbols
        if self.change_var_to is not None:
            adata.var.index = adata.var[self.change_var_to]
            
        if self.map_path is not None:
            
            if self.dataset_name == 'velmeshev':
                # NOTE: split to only ensembl ids...
                adata.var.index = adata.var.index.str.split('\\|').str[0]
                
            map_df = read_csv(os.path.join('data', self.map_path))
            adata = map_gene_symbols(adata, map_df)
            
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
        for method in methods:
            score_key = method.magnitude if method.magnitude is not None else method.specificity
            
            adata.uns[score_key] = \
                method.by_sample(adata, 
                                 groupby=self.groupby,
                                 use_raw=False,
                                 sample_key=self.sample_key,
                                 verbose=True,
                                 n_perms=None,
                                 inplace=False
                                 )
        
        adata.write_h5ad(os.path.join('data', 'interim', f"{self.dataset_name}_processed.h5ad"))
        
        return adata
    
    def dim_reduction_pipe(self, adata, use_gpu=True):    
    
        # methods to use
        scores = li.mt.show_methods()
        # in case a method is missing Magnitude Score, use Specificity Score
        scores['score_key'] = scores["Magnitude Score"].fillna(scores["Specificity Score"])
        # drop duplicated scores (expr_prod for NATMI & Connectome)
        scores = scores.drop_duplicates(subset=['Method Name', 'score_key'])
        scores = scores[['Method Name', 'score_key']]
        
        _dict_setup(adata, 'mofa_res')
        _dict_setup(adata, 'tensor_res')
        
        for score_key in scores['score_key']:
            print(f"Creating views with: {score_key}")

            # Note: I should save results - to avoid re-running the same things
            run_mofatalk(adata=adata,
                         score_key=score_key,
                         sample_key=self.sample_key, 
                         condition_key=self.condition_key,
                         dataset_name=self.dataset_name,
                         n_factors=self.n_factors,
                         gpu_mode=False) # NOTE: use_gpu is not passed
            
            run_tensor_c2c(adata=adata,
                           score_key=score_key, 
                           sample_key=self.sample_key,
                           condition_key=self.condition_key, 
                           dataset_name=self.dataset_name,
                           n_factors=self.n_factors,
                           use_gpu=use_gpu
                           )
        
        adata.write(os.path.join('data', 'results', f'{self.dataset_name}_dimred.h5ad'))
        
        
    def run_classifier(self, adata, n_estimators=100):
        """
        Run a Random Forest classifier on the given data and return performance metrics.
        """
        # TODO: avoid iterations over methods later on
        score_keys = adata.uns['mofa_res']['X'].keys()
        
        evaluate = []
        
        n_samples = adata.obs[self.sample_key].nunique()
        
        random_states = range(0, 5)
        
        for state in random_states:
        
            splits = _generate_splits(n_samples, random_state=state, n_factors=self.n_factors)
        
            for index, row in splits.iterrows():
                fold = row['fold']
                train_index = row['train']
                test_index = row['test']
                
                for score_key in score_keys:
                    mofa = adata.uns['mofa_res']
                    X_m = mofa['X_0'][score_key]
                    y_m = mofa['y_0'][score_key]

                    tensor = adata.uns['tensor_res']
                    X_t = tensor['X_0'][score_key]
                    y_t = tensor['y_0'][score_key]
            
                    assert all(np.isin(['mofa_res', 'tensor_res'], adata.uns_keys())), 'Run the setup function first.'
                    
                    assert X_m.shape[0] == X_t.shape[0], 'mofa and tensor have different number of samples.'
                    
                    # Evaluate MOFA
                    auroc, tpr, fpr, f1, oob_score = _run_rf_auc(X_m, y_m, train_index, test_index, n_estimators=n_estimators)
                    evaluate.append(_assign_dict(reduction_name='mofa', score_key=score_key, state=state, fold=fold,
                                                auroc=auroc, tpr=tpr, fpr=fpr, f1_score=f1, oob_score=oob_score,
                                                train_split=train_index, test_split=test_index, test_classes=y_m[test_index]))
                    
                    # Evaluate Tensor
                    auroc, tpr, fpr, f1, oob_score = _run_rf_auc(X_t, y_t, train_index, test_index, n_estimators=n_estimators)
                    evaluate.append(_assign_dict(reduction_name='tensor', score_key=score_key, state=state, fold=fold,
                                                auroc=auroc, tpr=tpr, fpr=fpr, f1_score=f1, oob_score=oob_score,
                                                train_split=train_index, test_split=test_index, test_classes=y_m[test_index]))
                fold += 1
                
        evaluate = pd.DataFrame(evaluate)
        evaluate['dataset'] = self.dataset_name
        adata.uns['evaluate'] = evaluate
        
        evaluate.to_csv(os.path.join('data', 'results', f'{self.dataset_name}.csv'), index=False)
