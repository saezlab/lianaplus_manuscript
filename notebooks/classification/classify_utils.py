import os
import gc

import numpy as np
import pandas as pd
import scanpy as sc

import muon as mu
import liana as li

import cell2cell as c2c
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 3
INVERSE_FUN = lambda x: -np.log2(x)

# TODO: run method -> classify; next method (not loop over all methods)

def _dict_setup(adata, uns_key):
    adata.uns[uns_key] = dict()
    adata.uns[uns_key] = {'X': {}, 'X_0': {}, 'y_0': {}, 'dimred_extra': {}}


def _encode_y(y):
    # create a LabelEncoder object & and transform the labels
    le = LabelEncoder()
    le.fit(y)
    return le.transform(y)


def _generate_splits(n_samples, random_state, n_factors):
    
    X_dummy = np.ones((n_samples, n_factors), dtype=np.int16)
    y_dummy = np.ones(n_samples, dtype=np.int16)
    
    splits = pd.DataFrame(columns=['fold', 'train', 'test'])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    fold = 0

    for train_index, test_index in skf.split(X_dummy, y_dummy):
        splits.loc[len(splits)] = [fold, train_index, test_index]

        print(f"{fold}: {train_index}, {test_index}")
        fold += 1
    
    # write splits
    return splits

def run_mofatalk(adata, score_key, sample_key, condition_key, dataset_name, n_factors, gpu_mode=False):

    mdata = li.multi.lrs_to_views(adata,
                                  sample_key=sample_key,
                                  score_key=score_key,
                                  inverse_fun=INVERSE_FUN,
                                  obs_keys=[condition_key], # add those to mdata.obs
                                  lr_prop = 0.33, # minimum required proportion of samples to keep an LR
                                  lrs_per_sample = 3, # minimum number of interactions to keep a sample in a specific view
                                  lrs_per_view = 10, # minimum number of interactions to keep a view
                                  samples_per_view = 5, # minimum number of samples to keep a view
                                  min_variance = 0, # minimum variance to keep an interaction
                                  lr_fill = 0, # fill missing LR values across samples with this
                                  verbose=True,
                                  uns_key=score_key
                                  ).copy()
    
    mu.tl.mofa(mdata,
               use_obs='union',
               convergence_mode='medium',
               n_factors=n_factors,
               outfile=os.path.join('data', 'results', 'models', dataset_name, f'{score_key}.hdf5'),
               seed=1337,
               gpu_mode=gpu_mode,
               )
    
    y = mdata.obs[condition_key]
    
    # save results
    factor_scores = li.ut.get_factor_scores(mdata, obsm_key='X_mofa').copy()
    adata.uns['mofa_res']['X'][score_key] = factor_scores
    
    ## create & write to dataset_name folder
    os.makedirs(os.path.join('data', 'results', 'mofa', dataset_name), exist_ok=True)
    factor_scores.to_csv(os.path.join('data', 'results', 'mofa', dataset_name, f'{score_key}.csv'))    
    
    adata.uns['mofa_res']['X_0'][score_key] = mdata.obsm['X_mofa'].copy()
    adata.uns['mofa_res']['y_0'][score_key] = _encode_y(y)
    
    gc.collect()
    


def run_tensor_c2c(adata, score_key, sample_key, condition_key, dataset_name, n_factors, use_gpu=True):
    if use_gpu:
        import tensorly as tl
        tl.set_backend('pytorch')
        device = 'cuda'
    else:
        device = 'cpu'
    
    tensor = li.multi.to_tensor_c2c(adata,
                                    sample_key = sample_key,
                                    inverse_fun = INVERSE_FUN,
                                    score_key = score_key, # can be any score from liana
                                    how='outer', # how to join the samples
                                    non_expressed_fill = 0, # value to fill non-expressed interactions
                                    outer_fraction = 0.33, 
                                    uns_key=score_key
                                    )
    
    context_dict = adata.obs[[sample_key, condition_key]].drop_duplicates()
    context_dict = dict(zip(context_dict[sample_key], context_dict[condition_key]))
    context_dict = defaultdict(lambda: 'Unknown', context_dict)

    tensor_meta = c2c.tensor.generate_tensor_metadata(interaction_tensor=tensor,
                                                    metadata_dicts=[context_dict, None, None, None],
                                                    fill_with_order_elements=True,
                                                    )
    
    output_folder = os.path.join('data', 'results', 'tensor', dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    tensor = c2c.analysis.run_tensor_cell2cell_pipeline(tensor,
                                                        tensor_meta,
                                                        copy_tensor=True, # Whether to output a new tensor or modifying the original
                                                        rank=n_factors, 
                                                        tf_optimization='regular', # To define how robust we want the analysis to be.
                                                        random_state=1337, # Random seed for reproducibility
                                                        device=device,
                                                        elbow_metric='error',
                                                        smooth_elbow=False, 
                                                        upper_rank=20,
                                                        tf_init='random', 
                                                        tf_svd='numpy_svd', 
                                                        cmaps=None, 
                                                        sample_col='Element',
                                                        group_col='Category', 
                                                        output_fig=False,
                                                        output_folder=output_folder
                                                        )
    
    factor_scores = tensor.factors['Contexts'].join(tensor_meta[0].set_index('Element'))
    y = factor_scores['Category']
    
    # save results TODO: change to dict somehow?
    adata.uns['tensor_res']['X'][score_key] = factor_scores.copy()
    
    ## write to dataset_name folder
    factor_scores.to_csv(os.path.join(output_folder, f'{score_key}.csv'))    
    
    adata.uns['tensor_res']['X_0'][score_key] = tensor.factors['Contexts'].values
    adata.uns['tensor_res']['y_0'][score_key] = _encode_y(y)
    
    gc.collect()
    
    
def _run_rf_auc(X, y, train_index, test_index, n_estimators=100):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # NOTE: I define it here, which means that if I pass the same splits, the same results will be returned.
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=1337, oob_score=True)
    clf.fit(X_train, y_train)
    
    oob_score = clf.oob_score_

    # NOTE: I'm using probabilities for AUC (more nuanced than binary predictions)
    y_prob = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auroc = auc(fpr, tpr)
    
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return auroc, tpr, fpr, f1, oob_score


def _assign_dict(reduction_name, score_key, state, fold, auroc, tpr, fpr, f1_score, oob_score, train_split, test_split, test_classes):
    return {'reduction_name': reduction_name,
            'score_key': score_key, 
            'state': state, 
            'fold': fold,
            'auroc': auroc, 
            'tpr': tpr,
            'fpr': fpr,
            'f1_score': f1_score, 
            'oob_score': oob_score,
            'train_split': train_split, 
            'test_split': test_split, 
            'test_classes' : test_classes
            }
