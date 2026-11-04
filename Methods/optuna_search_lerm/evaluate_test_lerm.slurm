#!/bin/bash
# One-shot TEST scoring for the LERM winner. Submit from ~/Syntheticdata/optimization_warming:
#   cd ~/Syntheticdata/optimization_warming && sbatch evaluate_test_lerm.slurm
# Needs evaluate_test_lerm.py next to optuna_search_lerm.py, and results_lerm/best_params.json.
#SBATCH --job-name=evaluate_test_lerm
#SBATCH --account=julieneg_1707
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=results_lerm/evaluate_test_lerm_%j.out
#SBATCH --error=results_lerm/evaluate_test_lerm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niati@usc.edu

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ammonyte_env
PYTHON=~/.conda/envs/ammonyte_env/bin/python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd ~/Syntheticdata/optimization_warming

$PYTHON -u evaluate_test_lerm.py \
    --outdir results_lerm \
    --test-path ~/Syntheticdata/train_test_split_data/test.npz \
    --n-jobs "$SLURM_CPUS_PER_TASK"
