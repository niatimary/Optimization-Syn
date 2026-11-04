#!/bin/bash
# Submit from ~/Syntheticdata/optimization_warming
# (`cd ~/Syntheticdata/optimization_warming && sbatch optuna_search_warming.slurm`).
# Both optimization_warming/logs/ and optimization_warming/results/ must exist
# before submission -- SLURM opens --output/--error at job start, before this
# script's own commands run, and this script does not create either for you.
#
# Layout expected:
#   ~/Syntheticdata/train_test_split_data/{train,validation,test}.npz  (data -- the NEW 30-seed split, replacing the old files)
#   ~/Syntheticdata/optimization_warming/{optuna_search.py,
#       method_configs.py, evaluate_test.py, optuna_search_warming.slurm,
#       logs/, results/}
#
# 24 cpus, 175G mem. Originally raised from 128G after job 12118812 was
# OOM-killed at trial #54, immediately following a Dynp trial -- Dynp has
# since been removed entirely from optuna_search.py (see its module
# docstring), which was the source of that memory pressure (every remaining
# method is penalty-based and far cheaper per call). 175G is kept anyway as a
# safe, already-verified ceiling, and a new --max-call-seconds=7200 (2h) hard
# per-series timeout (see optuna_search.py's CallTimeoutError) is now also in
# place as a general safety net against any single call hanging. Partition
# switched from oneweek to main (see conversation) -- main's 2-day time limit
# comfortably covers this run's --time below, and main has far more nodes /
# faster turnover than oneweek's small, heavily-contended pool.
#
# Budget: --n-trials below is a hard cap on TOTAL trials across every resumed
# invocation (not "N more" each time) -- the study stops once it's reached,
# possibly well before --time is up. --n-trials 400 is a hard cap sized to
# the discrete (method, cost, min_size) branches in the (now Dynp-free)
# search space (about 60 branches, so roughly 6-7 trials per branch) -- --min-trials/--patience are expected to stop the study
# well before that cap is reached in practice. --min-trials/--patience/
# --min-delta are the study-level early-stopping rule (see optuna_search.py's
# module docstring's 'Early stopping' section) -- this is the DELIBERATE
# point of comparison against optuna_search_pruning.py's WilcoxonPruner
# per-trial pruning, run as a separate study (results_pruning/) -- compare
# both best_params.json on validation once both have run.
# Early stopping here: stop after 80 trials without a >0.01 improvement, but not before 100 trials.
# The 1-day limit is met by --timeout 79200 (22 h), which leaves ~2 h for the final train/validation
# scoring. After this run, check results/optuna_convergence.png to see if the curve actually
# plateaued before stopping (rule is fine), was still improving (raise
# patience/min-trials/n-trials and rerun), or plateaued much earlier (safe to
# shrink patience next time). --timeout is kept as a secondary wall-clock
# safety net so the run saves/exits cleanly instead of getting killed
# mid-write if it somehow runs long; it's no longer the primary budget
# mechanism. If the --n-trials cap isn't reached in one submission for any
# reason, just resubmit this same script -- the study is checkpointed
# (--storage) and resumes from wherever it left off, no trials lost, and
# will only run enough more trials to reach the cap.
#SBATCH --job-name=optuna_search_warming
#SBATCH --account=julieneg_1707
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=175G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/optuna_search_warming_%j.out
#SBATCH --error=logs/optuna_search_warming_%j.err
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

# Prevent thread oversubscription (parallelism comes from joblib processes)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd ~/Syntheticdata/optimization_warming

$PYTHON -u optuna_search.py \
    --objective warming \
    --train-path ~/Syntheticdata/train_test_split_data/train.npz \
    --val-path ~/Syntheticdata/train_test_split_data/validation.npz \
    --outdir results \
    --study-name search_warming_noisier \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --n-trials 400 \
    --min-trials 100 \
    --patience 80 \
    --min-delta 0.01 \
    --timeout 79200

echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
