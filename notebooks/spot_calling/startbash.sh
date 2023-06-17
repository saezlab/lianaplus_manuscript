#!/bin/bash
#SBATCH -p single
#SBATCH -N 1
#SBATCH --mem=50000
#SBATCH --cpus-per-task 8
#SBATCH --time=48:00:00
#SBATCH --job-name="liana2"
#SBATCH --output=spot_calling.out
#SBATCH --mail-user=daniel.dimitrov@uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --chdir /net/data.isilon/ag-saez/bq_ddimitrov/Repos/liana2_manuscript/notebooks/spot_calling

python $1