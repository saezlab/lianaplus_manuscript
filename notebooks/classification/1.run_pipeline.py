import os
import warnings
from processer import DatasetHandler
warnings.filterwarnings('ignore') # silence endless anndata spam

file_path = os.path.dirname(os.path.abspath(__file__))
datasets = ["carraro", "velmeshev" ,"reichart", "kuppe", "habermann"]

def run_classification(dataset_name, use_gpu=True):
    os.chdir(file_path)
    # Create an instance of DatasetHandler
    handler = DatasetHandler(dataset_name=dataset_name)
    
    # Process the dataset
    # adata = handler.process_dataset()
    import scanpy as sc
    adata = sc.read_h5ad(os.path.join('data', 'interim', f'{dataset_name}_processed.h5ad'))
    
    # Perform dimensionality reduction using dim_reduction_pipe function
    handler.dim_reduction_pipe(adata, use_gpu=use_gpu)
    
    # Run classifier on the reduced dataset using run_classifier function
    handler.run_classifier(adata)
    print(adata.uns['evaluate'].head())

if __name__ == "__main__":    
    for dataset_name in datasets:
        run_classification(dataset_name)
