#!/bin/bash
# Submit from ~/Syntheticdata/optimization_warming
# (`cd ~/Syntheticdata/optimization_warming && sbatch optuna_search_lerm_warming.slurm`).
# results_lerm_tau/ must exist before submission (`mkdir -p results_lerm_tau`) -- SLURM opens
# --output/--error at job start, before this script's own commands run. Logs and Optuna
# outputs both go to results_lerm_tau/, like results_ks/ for the KS test.
#
# Layout expected:
#   ~/Syntheticdata/train_test_split_data/{train,validation,test}.npz  (same data as the KS test)
#   ~/Syntheticdata/optimization_warming/{optuna_search_lerm.py,
#       optuna_search_lerm_warming.slurm, results_lerm_tau/}
#
# LERM (ammonyte TimeEmbeddedSeries -> find_epsilon -> laplacian_eigenmaps -> smooth_series
# -> lerm_transitions) cost per series is dominated by the eigendecomposition in
# laplacian_eigenmaps: measured ~20-90 s per series on 6001-point series (worst corner of the
# search space, m=15 / w_incre=2, was ~20-46 s), plus a few s for the 5-cell block_size
# sweep that reuses the same Fisher-information series. Rough budget at --n-trials 250:
#   250 trials x 60 series x ~40 s / 24 cores ~= 7h
# so the 2-day walltime below is generous headroom, kept the same as the other Optuna scripts.
#
# --n-trials is a hard cap on TOTAL trials across every resumed invocation. If a run is
# interrupted (walltime, OOM), resubmit this same script -- the study resumes from study.db.
# --max-call-seconds (default 7200 s) is a safety net against a hung series, not a cost limit.
#SBATCH --job-name=optuna_search_lerm_warming
#SBATCH --account=julieneg_1707
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=175G
#SBATCH --time=2-00:00:00
#SBATCH --output=results_lerm_tau/optuna_search_lerm_warming_%j.out
#SBATCH --error=results_lerm_tau/optuna_search_lerm_warming_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niati@usc.edu

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ammonyte_env

PYTHON=~/.conda/envs/ammonyte_env/bin/python

$PYTHON -c "import ammonyte; print(f'ammonyte {ammonyte.__version__} loaded')" || {
    echo "ERROR: ammonyte not found. Run: pip install -e ~/Ammonyte"
    exit 1
}

# Prevent thread oversubscription (parallelism comes from joblib processes). Important here:
# without this, each worker's eigendecomposition spawns its own BLAS threads and the 24 workers
# slow each other down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# PyRQA (used by find_epsilon) needs an OpenCL platform; compute nodes have none, so use the
# conda pocl CPU runtime. Without the vendors path every series fails with
# "clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR". One pocl thread per worker process.
export OCL_ICD_VENDORS=$CONDA_PREFIX/etc/OpenCL/vendors
export POCL_MAX_CPU_CU_COUNT=1     # older pocl
export POCL_MAX_PTHREAD_COUNT=1    # pocl >= 4 (7.1 installed)
$PYTHON -c "import pyopencl as cl; print('OpenCL platforms:', cl.get_platforms())" || {
    echo "ERROR: no OpenCL platform (pocl) visible"
    exit 1
}

cd ~/Syntheticdata/optimization_warming

for f in train validation; do
    [ -f ~/Syntheticdata/train_test_split_data/$f.npz ] || {
        echo "ERROR: missing ~/Syntheticdata/train_test_split_data/$f.npz"
        exit 1
    }
done

$PYTHON -u optuna_search_lerm.py \
    --objective warming \
    --train-path ~/Syntheticdata/train_test_split_data/train.npz \
    --val-path ~/Syntheticdata/train_test_split_data/validation.npz \
    --outdir results_lerm_tau \
    --study-name search_lerm_tau_warming \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --n-trials 250 \
    --min-trials 100 \
    --patience 50 \
    --min-delta 0.01 \
    --timeout 165000

echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
