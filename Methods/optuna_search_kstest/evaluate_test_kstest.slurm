#!/bin/bash
# One-shot TEST scoring for the KS-test winner. Submit from ~/Syntheticdata/optimization_warming:
#   cd ~/Syntheticdata/optimization_warming && sbatch evaluate_test_kstest.slurm
# Needs evaluate_test_kstest.py next to optuna_search_kstest.py, and results_ks/best_params.json.
#SBATCH --job-name=evaluate_test_kstest
#SBATCH --account=julieneg_1707
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=results_ks/evaluate_test_kstest_%j.out
#SBATCH --error=results_ks/evaluate_test_kstest_%j.err
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

$PYTHON -u evaluate_test_kstest.py \
    --outdir results_ks \
    --test-path ~/Syntheticdata/train_test_split_data/test.npz \
    --n-jobs "$SLURM_CPUS_PER_TASK"
