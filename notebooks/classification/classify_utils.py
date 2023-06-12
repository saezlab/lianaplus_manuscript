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
N_FACTORS = 10
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


def _generate_splits(n_samples, random_state):
    
    X_dummy = np.ones((n_samples, N_FACTORS), dtype=np.int16)
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
    


# def dim_reduction_pipe(adata, dataset_name, sample_key, condition_key, batch_key, use_gpu=True):    
    
#     # methods to use
#     methods = li.mt.show_methods()
#     # in case a method is missing Magnitude Score, use Specificity Score
#     methods['score_key'] = methods["Magnitude Score"].fillna(methods["Specificity Score"])
#     # remove Geometric Mean	method
#     methods = methods[methods['Method Name'] != 'Geometric Mean']
#     # drop duplicated scores (expr_prod for NATMI & Connectome)
#     methods = methods.drop_duplicates(subset=['Method Name', 'score_key'])
#     methods = methods[['Method Name', 'score_key']]
    
#     _dict_setup(adata, 'mofa_res')
#     _dict_setup(adata, 'tensor_res')
    
#     for score_key in methods['score_key']:
#         print(f"Creating views with: {score_key}")

#         # Note: I should save results - to avoid re-running the same things
#         run_mofatalk(adata=adata, score_key=score_key, sample_key=sample_key, 
#                      condition_key=condition_key, batch_key=batch_key, dataset_name=dataset_name,
#                      gpu_mode=False) # NOTE: use_gpu is not passed
        
#         run_tensor_c2c(adata=adata, score_key=score_key, sample_key=sample_key,
#                        condition_key=condition_key, dataset_name=dataset_name, use_gpu=use_gpu)
    
#     adata.write(os.path.join('data', 'results', f'{dataset_name}_dimred.h5ad'))


def run_mofatalk(adata, score_key, sample_key, condition_key, batch_key, dataset_name, gpu_mode=False):

    mdata = li.multi.lrs_to_views(adata,
                                  sample_key=sample_key,
                                  score_key=score_key,
                                  inverse_fun=INVERSE_FUN,
                                  obs_keys=[condition_key, batch_key], # add those to mdata.obs
                                  lr_prop = 0.33, # minimum required proportion of samples to keep an LR
                                  lrs_per_sample = 5, # minimum number of interactions to keep a sample in a specific view
                                  lrs_per_view = 15, # minimum number of interactions to keep a view
                                  samples_per_view = 5, # minimum number of samples to keep a view
                                  min_variance = 0, # minimum variance to keep an interaction
                                  lr_fill = 0, # fill missing LR values across samples with this
                                  verbose=True
                                  ).copy()
    
    mu.tl.mofa(mdata,
               use_obs='union',
               convergence_mode='medium',
               n_factors=N_FACTORS,
               seed=1337,
               gpu_mode=gpu_mode,
               )
    
    y = mdata.obs[condition_key]
    
    # save results
    factor_scores = li.multi.get_factor_scores(mdata, obsm_key='X_mofa').copy()
    adata.uns['mofa_res']['X'][score_key] = factor_scores
    
    ## create & write to dataset_name folder
    os.makedirs(os.path.join('data', 'results', 'mofa', dataset_name), exist_ok=True)
    factor_scores.to_csv(os.path.join('data', 'results', 'mofa', dataset_name, f'{score_key}.csv'))    
    
    adata.uns['mofa_res']['X_0'][score_key] = mdata.obsm['X_mofa'].copy()
    adata.uns['mofa_res']['y_0'][score_key] = _encode_y(y)
    
    gc.collect()
    


def run_tensor_c2c(adata, score_key, sample_key, condition_key, dataset_name, use_gpu=True):
    if use_gpu:
        import tensorly as tl
        tl.set_backend('pytorch')
        device = 'cuda'
    else:
        device = 'cpu'
    
    tensor = li.multi.to_tensor_c2c(adata,
                                    sample_key=sample_key,
                                    inverse_fun=INVERSE_FUN,
                                    score_key=score_key, # can be any score from liana
                                    how='outer', # how to join the samples
                                    non_expressed_fill=0, # value to fill non-expressed interactions
                                    outer_fraction = 0.33, 
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
                                                        rank=N_FACTORS, 
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
                                                        )
    
    factor_scores = tensor.factors['Contexts'].join(tensor_meta[0].set_index('Element'))
    y = factor_scores['Category']
    
    # save results TODO: change to dict somehow?
    adata.uns['tensor_res']['X'][score_key] = factor_scores.copy()
    
    ## create & write to dataset_name folder
    os.makedirs(os.path.join('data', 'results', 'tensor', dataset_name), exist_ok=True)
    factor_scores.to_csv(os.path.join('data', 'results', 'tensor', dataset_name, f'{score_key}.csv'))    
    
    adata.uns['tensor_res']['X_0'][score_key] = tensor.factors['Contexts'].values
    adata.uns['tensor_res']['y_0'][score_key] = _encode_y(y)
    
    gc.collect()


# def run_classifier(adata, dataset_name, n_estimators=100):
#     """
#     Run a Random Forest classifier on the given data and return performance metrics.
#     """
#     # TODO: avoid iterations over methods later on
#     score_keys = adata.uns['mofa_res']['X'].keys()
    
#     evaluate = []
    
#     n_samples = adata.obs[adata.uns['sample_key']].nunique()
    
#     random_states = range(0, 5)
    
#     for state in random_states:
    
#         splits = _generate_splits(n_samples, random_state=state)
    
#         for index, row in splits.iterrows():
#             fold = row['fold']
#             train_index = row['train']
#             test_index = row['test']
            
#             for score_key in score_keys:
#                 mofa = adata.uns['mofa_res']
#                 X_m = mofa['X_0'][score_key]
#                 y_m = mofa['y_0'][score_key]

#                 tensor = adata.uns['tensor_res']
#                 X_t = tensor['X_0'][score_key]
#                 y_t = tensor['y_0'][score_key]
        
#                 assert all(np.isin(['mofa_res', 'tensor_res'], adata.uns_keys())), 'Run the setup function first.'
                
#                 assert X_m.shape[0] == X_t.shape[0], 'mofa and tensor have different number of samples.'
                
#                 # Evaluate MOFA
#                 auroc, tpr, fpr, f1, oob_score = _run_rf_auc(X_m, y_m, train_index, test_index, n_estimators=n_estimators)
#                 evaluate.append(_assign_dict(reduction_name='mofa', score_key=score_key, state=state, fold=fold,
#                                              auroc=auroc, tpr=tpr, fpr=fpr, f1_score=f1, oob_score=oob_score,
#                                              train_split=train_index, test_split=test_index, test_classes=y_m[test_index]))
                
#                 # Evaluate Tensor
#                 auroc, tpr, fpr, f1, oob_score = _run_rf_auc(X_t, y_t, train_index, test_index, n_estimators=n_estimators)
#                 evaluate.append(_assign_dict(reduction_name='tensor', score_key=score_key, state=state, fold=fold,
#                                              auroc=auroc, tpr=tpr, fpr=fpr, f1_score=f1, oob_score=oob_score,
#                                              train_split=train_index, test_split=test_index, test_classes=y_m[test_index]))
#             fold += 1
            
#     evaluate = pd.DataFrame(evaluate)
#     evaluate['dataset'] = dataset_name
#     adata.uns['evaluate'] = evaluate
    
#     evaluate.to_csv(os.path.join('data', 'results', f'{dataset_name}.csv'), index=False)
    
    
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