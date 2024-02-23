import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_squared_error

import scanpy as sc
from mudata import MuData
import liana as li
import stlearn as st
from anndata import AnnData

def run_local(adata, function_name, standardize , **kwargs):
    li.ut.spatial_neighbors(adata, set_diag=True, bandwidth=150, cutoff=0.1, standardize=standardize)
    li.mt.lr_bivar(adata,
                function_name=function_name,
                obsm_added=function_name, 
                use_raw=False, 
                **kwargs
                )

lrs = li.rs.explode_complexes(li.rs.select_resource())
lrs['interaction'] = lrs['ligand'] + '_' + lrs['receptor']

def run_stlearn(adata, key='-log10(p_adjs)', obsm_key='stLearn'):
    st.tl.cci.run(adata, 
        np.unique(lrs['interaction'].values),
        min_spots = 20, 
        distance=None, # None defaults to spot+immediate neighbours; distance=0 for within-spot mode
        n_pairs=100, # Number of random pairs to generate; low as example, recommend ~10,000
        n_cpus=None, # Number of CPUs for parallel. If None, detects & use all available.
        )
    lr_info = adata.uns['lr_summary']
    scores = AnnData(var=pd.DataFrame(index=lr_info.index),
                     obs=adata.obs,
                     X=adata.obsm[key], 
                     uns=adata.uns, 
                     obsm=adata.obsm)
    adata.obsm[obsm_key] = scores
    

def convert_scanpy(adata, use_quality: str = "hires"):
    """
    Taken from stLearn https://github.com/BiomedicalMachineLearning/stLearn
    """

    adata.var_names_make_unique()

    library_id = list(adata.uns["spatial"].keys())[0]

    if use_quality == "fulres":
        image_coor = adata.obsm["spatial"]
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + use_quality + "_scalef"
        ]
        image_coor = adata.obsm["spatial"] * scale

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = use_quality

    return adata

clf = RandomForestClassifier(n_estimators=100, random_state=1337, oob_score=True, n_jobs=-1)

def _evaluate_classifier(X, y, train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    
    oob_score = clf.oob_score_

    # NOTE: I'm using probabilities for AUC (more nuanced than binary predictions)
    y_prob = clf.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return roc_auc, f1, oob_score


def _get_assay(adata, function_name):
    if isinstance(adata, MuData):
        X = adata.mod[function_name].X
    else:
        X = adata.obsm[function_name].X
        
    return X
        
def run_rf_auc(adata, dataset_name):
    X_dummy = adata.X
    y_dummy = adata.obs['spot_label'].values
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
    function_names = adata.uns['function_names']

    performance = pd.DataFrame(columns=['dataset_name', 'function_name',
                                        'roc_auc', 'f1', 'oob_score',
                                        #'train_index', 'test_index'
                                        ])
    fold = 0

    for train_index, test_index in skf.split(X_dummy, y_dummy):
        print(f"Evaluating {dataset_name}; Fold: {fold}")
        
        for function_name in function_names:
            X = _get_assay(adata, function_name)
            y = adata.obs['spot_label'].values
            
            roc_auc, f1, oob_score =  _evaluate_classifier(X, y, train_index, test_index)
            
            performance.loc[len(performance)] = [dataset_name, function_name,
                                                 roc_auc, f1, oob_score,
                                                 #train_index, test_index
                                                 ]

        fold += 1
    
    adata.uns['performance'] = performance


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


def _evaluate_regression(X, y, dataset_name, function_name, regressor):
    # Create a KFold cross-validator with 5 splits
    kfold = KFold(n_splits=5, shuffle=True, random_state=1337)

    # Initialize lists to store R2 scores and RMSE values
    r2_scores = []
    rmse_scores = []
    
    # Perform cross-validation
    for train_index, test_index in kfold.split(X):
        # Split the data into training and testing sets for each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit the regressor on the training data
        regressor.fit(X_train, y_train)
        
        # Make predictions on the testing data
        y_pred = regressor.predict(X_test)
        
        # Calculate R2 score
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)
        print(f"test_index: {test_index}")
        
    eval_df = pd.DataFrame({'dataset_name':dataset_name.split('.')[0],
                            'function_name':function_name, 
                            'r2': r2_scores, 'rmse': rmse_scores})

    return eval_df
