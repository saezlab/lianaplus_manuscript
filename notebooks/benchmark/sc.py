from benchmark import _benchmark, _sample_anndata, _sample_resource, obs_range, n_times

import pandas as pd
import liana as li
import gc

from os import path
file_path = path.dirname(path.abspath(__file__))

# Parameters to pass to the benchmark function
methods = {
    "CellChat": li.mt.cellchat,
    "Connectome": li.mt.connectome,
    "CellPhoneDB": li.mt.cellphonedb,
    "Geometric Mean": li.mt.geometric_mean,
    "NATMI": li.mt.natmi,
    "Rank Aggregate": li.mt.rank_aggregate,
    "SingleCellSignalR": li.mt.singlecellsignalr,
    "log2FC": li.mt.logfc,
    "scSeqComm": li.mt.scseqcomm,
    }


def benchmark_methods(obs_range, methods, output_file, n_times=5, n_lrs=2000, **kwargs):
    benchmark_stats = pd.DataFrame(columns=["method", "dataset", "time", "memory"])

    for n_obs in obs_range:
        print(n_obs)
        adata = _sample_anndata(n_obs=n_obs)
        resource = _sample_resource(adata, n_lrs=n_lrs)
            
        for method in methods.keys():
            for _ in range(n_times):
                print(method)
                time, memory = _benchmark(function=methods[method],
                                          # kwargs:
                                          adata=adata,
                                          resource=resource,
                                          **kwargs
                                          )
                benchmark_stats.loc[len(benchmark_stats)] = [method, n_obs, time, memory]
        benchmark_stats.to_csv(output_file, index=False)
        
        gc.collect()

    return benchmark_stats

benchmark_stats = benchmark_methods(obs_range=obs_range,
                                    methods=methods,
                                    output_file=path.join(file_path, 'sc_stats.csv'),
                                    groupby='cell_type',
                                    n_times=n_times,
                                    n_lrs=2000,
                                    use_raw=False,
                                    verbose=False,
                                    n_jobs=4
                                    )