import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression # TODO: replace with RF

from utils import load_prep_slide, _evaluate_regression, _get_function_names

import liana as li

data_dir = os.path.join('..', '..', 'data', 'heart_visium')

# scan names of all datasets
dataset_names = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
function_names = _get_function_names()


# Initialize the Random Forest Regressor with default parameters
# regressor = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=1337)
regressor = LinearRegression() # TODO: replace with RF

results = []


for dataset_name in dataset_names:

    print(f'Loading {dataset_name}')
    # Load and preprocess data
    adata = load_prep_slide(data_dir, dataset_name)
        
    for function_name in function_names:
        
        li.mt.lr_bivar(adata,
                       function_name=function_name,
                       expr_prop=0.1,
                       pvalue_method=None, 
                       use_raw=False,
                       )
        
        print(f'Running {function_name}')
        
        y = adata.obsm['compositions'].values
        X = adata.obsm['local_scores'].values
        
        # evaluate
        eval_df = _evaluate_regression(X, y, dataset_name, function_name, regressor)
        results.append(eval_df)
        
results = pd.concat(results)
results.to_csv('results.csv', index=False)
