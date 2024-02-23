import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from utils import load_prep_slide, _evaluate_regression, run_local, run_stlearn, convert_scanpy

import liana as li

data_dir = os.path.join('data', 'heart_visium')
file_path = os.path.dirname(os.path.abspath(__file__))

# scan names of all datasets
dataset_names = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
function_names = function_names = li.mt.bivar.show_functions()['name'].values
function_names = np.insert(function_names, 0, 'stLearn')

# Initialize the Random Forest Regressor with default parameters
regressor = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=1337)

results = []

for dataset_name in dataset_names:

    print(f'Loading {dataset_name}')
    # Load and preprocess data
    adata = load_prep_slide(data_dir, dataset_name)
    
    # NOTE: stLearn specific
    adata = convert_scanpy(adata)
        
    for function_name in function_names:
        
        if function_name == 'stLearn':
            run_stlearn(adata)
        else:
            if function_name not in ['product', 'norm_product']:
                standardize = False
            else:
                standardize = True
            
            run_local(adata, function_name, standardize=standardize)
        
        print(f'Running {function_name}')
        
        y = adata.obsm['compositions'].values
        X = adata.obsm[function_name].X
        
        # evaluate
        eval_df = _evaluate_regression(X, y, dataset_name, function_name, regressor)
        results.append(eval_df)
    
    # save preliminary results    
    pd.concat(results).to_csv(os.path.join(file_path, 'regression_results.csv'), index=False)

results = pd.concat(results)
results.to_csv(os.path.join(file_path, 'regression_results.csv'), index=False)
