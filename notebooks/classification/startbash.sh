#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100000
#SBATCH --cpus-per-task 2
#SBATCH --time=5:30:00
#SBATCH --job-name="liana2"
#SBATCH --output=liana2.out
#SBATCH --mail-user=daniel.dimitrov@uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --chdir /net/data.isilon/ag-saez/bq_ddimitrov/Repos/liana2_manuscript/notebooks/classification

## loop over all datasets
datasets=("carraro" "kuppe" "habermann" "velmeshev" "reichart")  # List of datasets

for dataset in "${datasets[@]}"; do
    echo "Now running dataset: $dataset"
    python 1.run_pipeline.py "$dataset"
done