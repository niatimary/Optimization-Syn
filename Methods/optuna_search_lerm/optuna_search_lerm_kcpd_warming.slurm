#!/bin/bash
# Submit from ~/Syntheticdata/optimization_warming
# (`cd ~/Syntheticdata/optimization_warming && mkdir -p results_lerm_kcpd && sbatch optuna_search_lerm_kcpd_warming.slurm`).
# results_lerm_kcpd/ must exist before submission -- SLURM opens --output/--error at job start.
#
# Layout expected:
#   ~/Syntheticdata/train_test_split_data/{train,validation,test}.npz
#   ~/Syntheticdata/optimization_warming/{optuna_search_lerm_kcpd.py,
#       optuna_search_lerm_kcpd_warming.slurm, results_lerm_kcpd/}
#
# LERM with KernelCPD, everything tuned together (see the VERSION 2 note in the script):
#   tuned by Optuna: m [2,12], tau [2,48] log, density [0.01,0.15] log, w_size [10,60], w_incre_frac [0.1,0.5]
#   swept inside each trial: k in {0.05,0.1,0.15,0.25,0.35,0.5,0.75,1} in
#       beta = k * 8.3475 * log(n_FI) / log(6001)
#   fixed: block_size 2; linear kernel, min_size 20 raw samples, jump 1.
# Cost per trial is the same band as optuna_search_lerm_kcpd (eigendecomposition dominates; the 8-value k
# sweep reuses one FI series). Known issue from the tau-tuned bounds run: ~16% of trials failed with
# "RuntimeError: Runaway operation" at large m; those trials are marked FAIL and skipped (the study stops
# only after 5 consecutive failures).
#
# --n-trials is a hard cap on TOTAL trials across every resumed invocation. If interrupted, resubmit
# this same script -- the study resumes from study.db.
#SBATCH --job-name=optuna_search_lerm_kcpd_warming
#SBATCH --account=julieneg_1707
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=175G
#SBATCH --time=2-00:00:00
#SBATCH --output=results_lerm_kcpd/optuna_search_lerm_kcpd_warming_%j.out
#SBATCH --error=results_lerm_kcpd/optuna_search_lerm_kcpd_warming_%j.err
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

$PYTHON -u optuna_search_lerm_kcpd.py \
    --objective warming \
    --train-path ~/Syntheticdata/train_test_split_data/train.npz \
    --val-path ~/Syntheticdata/train_test_split_data/validation.npz \
    --outdir results_lerm_kcpd \
    --study-name search_lerm_kcpd_warming \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --n-trials 250 \
    --min-trials 100 \
    --patience 50 \
    --min-delta 0.01 \
    --timeout 165000

echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
