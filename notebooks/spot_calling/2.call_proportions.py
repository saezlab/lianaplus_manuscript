import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from utils import load_prep_slide, _evaluate_regression

import liana as li

data_dir = os.path.join('data', 'heart_visium')
file_path = os.path.dirname(os.path.abspath(__file__))

# scan names of all datasets
dataset_names = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
function_names = function_names = li.mt.bivar.show_functions()['name'].values

# Initialize the Random Forest Regressor with default parameters
regressor = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=1337)

results = []

for dataset_name in dataset_names:

    print(f'Loading {dataset_name}')
    # Load and preprocess data
    adata = load_prep_slide(data_dir, dataset_name)
        
    for function_name in function_names:
        
        li.mt.lr_bivar(adata,
                       function_name=function_name,
                       expr_prop=0.1,
                       n_perms=None, 
                       use_raw=False,
                       )
        
        print(f'Running {function_name}')
        
        y = adata.obsm['compositions'].values
        X = adata.obsm['local_scores'].X
        
        # evaluate
        eval_df = _evaluate_regression(X, y, dataset_name, function_name, regressor)
        results.append(eval_df)
    
    # save preliminary results    
    pd.concat(results).to_csv(os.path.join(file_path, 'results.csv'), index=False)

results = pd.concat(results)
results.to_csv(os.path.join(file_path, 'results.csv'), index=False)
