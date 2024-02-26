import pandas as pd
from benchmark import _benchmark, _sample_anndata, _sample_resource, obs_range, n_times
import liana as li
import gc

from os import path
file_path = path.dirname(path.abspath(__file__))

benchmark_stats = pd.DataFrame(columns=["method", "dataset", "time", "memory"])
function_names = li.mt.bivar.show_functions()['name'].values
n_lrs = 1000

for n_obs in obs_range:
    print(n_obs)
    
    adata = _sample_anndata(n_obs=n_obs)
    resource = _sample_resource(adata, n_lrs=n_lrs)
        
    for method in function_names:
        print(method)
        if (method=='masked_spearman') and (n_obs>50000):
            continue
        
        for _ in range(n_times):
            time, memory = _benchmark(function=li.mt.lr_bivar, 
                                      adata=adata, 
                                      function_name=method, 
                                      n_perms=None,
                                      use_raw=False, 
                                      verbose=False,
                                      mask_negatives=False,
                                      resource=resource
                                      )
            benchmark_stats.loc[len(benchmark_stats)] = [method, n_obs, time, memory]
            gc.collect()
    benchmark_stats.to_csv(path.join(file_path, "sp_stats.csv"), index=False)
