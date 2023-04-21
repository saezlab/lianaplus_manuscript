import numpy as np
import pandas as pd

import muon as mu
import liana as li

import cell2cell as c2c
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder


class NestedDict(dict):
    def __missing__(self, x):
        self[x] = NestedDict()
        return self[x]
    
    def __getattr__(self, x):
        return self[x]

def _encode_y(y):
    # create a LabelEncoder object & and transform the labels
    le = LabelEncoder()
    le.fit(y)
    return le.transform(y)



def run_mofatalk(adata, score_key, sample_key, condition_key, batch_key):
    mdata = li.multi.lrs_to_views(adata,
                                sample_key=sample_key,
                                score_key=score_key,
                                obs_keys=[condition_key, batch_key], # add those to mdata.obs
                                lr_prop = 0.2, # minimum required proportion of samples to keep an LR
                                lrs_per_sample = 5, # minimum number of interactions to keep a sample in a specific view
                                lrs_per_view = 10, # minimum number of interactions to keep a view
                                samples_per_view = 5, # minimum number of samples to keep a view
                                min_variance = 0, # minimum variance to keep an interaction
                                lr_fill = 0, # fill missing LR values across samples with this
                                verbose=True
                                ).copy()
    
    mu.tl.mofa(mdata,
               use_obs='union',
               convergence_mode='medium',
               n_factors=10,
               seed=1337
               )
    
    y = mdata.obs[condition_key]
    
    # save results
    adata.uns['mofa_res']['X'][score_key] = li.multi.get_factor_scores(mdata, obsm_key='X_mofa').copy()
    adata.uns['mofa_res']['y'][score_key] = y
    
    adata.uns['mofa_res']['X_0'][score_key] = mdata.obsm['X_mofa'].copy()
    adata.uns['mofa_res']['y_0'][score_key] = _encode_y(y)
    


def run_tensor_c2c(adata, score_key, sample_key, condition_key):
    
    tensor = li.multi.to_tensor_c2c(adata,
                                    sample_key=sample_key,
                                    score_key='magnitude_rank', # can be any score from liana
                                    how='outer', # how to join the samples
                                    non_expressed_fill=0, # value to fill non-expressed interactions
                                    outer_fraction = 0.2, 
                                    )
    
    context_dict = adata.obs[[sample_key, condition_key]].drop_duplicates()
    context_dict = dict(zip(context_dict[sample_key], context_dict[condition_key]))
    context_dict = defaultdict(lambda: 'Unknown', context_dict)

    tensor_meta = c2c.tensor.generate_tensor_metadata(interaction_tensor=tensor,
                                                    metadata_dicts=[context_dict, None, None, None],
                                                    fill_with_order_elements=True,
                                                    )
    
    
    tensor = c2c.analysis.run_tensor_cell2cell_pipeline(tensor,
                                                    tensor_meta,
                                                    copy_tensor=True, # Whether to output a new tensor or modifying the original
                                                    rank=10, 
                                                    tf_optimization='regular', # To define how robust we want the analysis to be.
                                                    random_state=0, # Random seed for reproducibility
                                                    device='cpu',
                                                    elbow_metric='error',
                                                    smooth_elbow=False, 
                                                    upper_rank=20,
                                                    tf_init='random', 
                                                    tf_svd='numpy_svd', 
                                                    cmaps=None, 
                                                    sample_col='Element',
                                                    group_col='Category', 
                                                    output_fig=False,
                                                    )
    
    factor_scores = tensor.factors['Contexts'].join(tensor_meta[0].set_index('Element'))
    y = factor_scores['Category']
    
    # save results TODO: change to dict somehow?
    adata.uns['tensor_res']['X'][score_key] = factor_scores.copy()
    adata.uns['tensor_res']['y'][score_key] = y
    
    adata.uns['tensor_res']['X_0'][score_key] = tensor.factors['Contexts'].values
    adata.uns['tensor_res']['y_0'][score_key] = _encode_y(y)



def run_classifier(adata, score_key, reduction_name, skf, n_estimators=500):
    """
    Run a Random Forest classifier on the given data and return the AUC.
    """
    
    reduction = adata.uns[reduction_name]
    
    X = reduction.X_0[score_key]
    y = reduction.y_0[score_key]
    
    fold = 0
    
    for train_index, test_index in skf.split(X, y):
        # NOTE: this does not ensure that the same samples are used for training and testing of MOFA and C2C
        # Question: is this a problem? Should we use the same samples for both?
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        roc_auc, tpr, fpr = _run_rf_auc(X_train, X_test, y_train, y_test, n_estimators=n_estimators)
        adata.uns['auc'].loc[len(adata.uns['auc'])] = [reduction_name, score_key, fold, roc_auc, tpr, fpr, train_index, test_index]
        
        fold += 1
    
    
def _run_rf_auc(X_train, X_test, y_train, y_test, n_estimators=500):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, tpr, fpr
