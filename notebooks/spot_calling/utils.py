import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_squared_error

import scanpy as sc
from mudata import MuData
import liana as li


clf = RandomForestClassifier(n_estimators=100, random_state=1337, oob_score=True)

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
        X = adata.obsm[function_name].values
        
    return X
        
def run_rf_auc(adata, dataset_name):
    
    # just need those to define dimensions
    X_dummy = _get_assay(adata, 'cosine')
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




def load_prep_slide(path, slide, add_sample_name=False, min_genes = 400):
    adata = sc.read_h5ad(os.path.join(path, slide))
    
    if add_sample_name:
        indeces = [f"{slide.split('.')[0]}-{i}" for i in adata.obs.index]
        adata.obs.index = indeces
        adata.obsm['compositions'].index = indeces
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=5)
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    li.ut.spatial_neighbors(adata, bandwidth=100, cutoff=0.1, set_diag=True)
    
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
        
        # train & test index
        print(f"train_index: {train_index}")
        print(f"test_index: {test_index}")
        
    eval_df = pd.DataFrame({'dataset_name':dataset_name.split('.')[0], 'function_name':function_name, 'r2': r2_scores, 'rmse': rmse_scores})

    return eval_df


def _get_function_names():
    function_names = li.mt.sp.show_functions()['name']
    function_names = list(function_names[~function_names.str.contains('masked')]) + ['masked_spearman']
    return function_names
