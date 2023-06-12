import sys
from processer import DatasetHandler

def main(dataset_name, use_gpu=True):
    # Create an instance of DatasetHandler
    handler = DatasetHandler(dataset_name=dataset_name)
    
    # Process the dataset
    adata = handler.process_dataset()
    
    # Perform dimensionality reduction using dim_reduction_pipe function
    handler.dim_reduction_pipe(adata, use_gpu=use_gpu)
    
    # Run classifier on the reduced dataset using run_classifier function
    handler.run_classifier(adata)
    
    print(adata.uns['evaluate'].head())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the dataset name as a command-line argument.")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    main(dataset_name)