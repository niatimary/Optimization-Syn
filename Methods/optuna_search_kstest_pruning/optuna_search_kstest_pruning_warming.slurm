#!/bin/bash
# Submit from ~/Syntheticdata/optimization_warming
# (`cd ~/Syntheticdata/optimization_warming && sbatch optuna_search_kstest_pruning_warming.slurm`).
# Both optimization_warming/logs/ and optimization_warming/results_ks_pruning/ must
# exist before submission -- SLURM opens --output/--error at job start, before
# this script's own commands run, and this script does not create either for you.
#
# Layout expected (this same optimization_warming/ directory also already holds
# optuna_search.py/optuna_search_pruning.py, the ruptures scripts, with their own
# results/ and results_pruning/ -- results_ks/ and results_ks_pruning/ below are
# named to not collide with those):
#   ~/Syntheticdata/train_test_split_data/{train,validation,test}.npz  (data)
#   ~/Syntheticdata/optimization_warming/{optuna_search_kstest.py, results_ks/,
#       optuna_search_kstest_pruning.py, optuna_search_kstest_pruning_warming.slurm,
#       logs/, results_ks_pruning/}
#
# WilcoxonPruner variant of optuna_search_kstest_warming.slurm (patience-based
# early stopping, in the sibling optuna_search_kstest/ folder, untouched) --
# runs optuna_search_kstest_pruning.py with its own --outdir (results_ks_pruning,
# NOT results_ks, NOT results/results_pruning (those belong to the ruptures
# scripts sharing this same directory) -- every script submitted from this same
# optimization_warming/ directory needs a distinct outdir, or they'd overwrite
# each other's optuna_trials.csv/best_params.json/study.db/etc; matches
# optuna_search_pruning_warming.slurm's own results/ vs. results_pruning/
# convention, one level further since results/results_pruning were already
# taken here) so results, logs, and the Optuna study never mix with the
# patience run's. Compare results_ks_pruning/best_params.json here against
# optuna_search_kstest/results_ks/best_params.json on validation before
# trusting either for TEST.
#
# KS test (ammonyte.Series.kstest / KS_test, Bagniewski et al. 2021) is
# roughly 500-1000x more expensive per call than the ruptures methods this
# project's other Optuna scripts tune: measured 12-22s per KS_test call on
# this project's actual train.npz (60 series, 6001 points each), versus
# ruptures' ~0.02-0.03s. This is exactly where WilcoxonPruner should matter
# more than it did for ruptures -- killing a clearly-bad trial after 1-2
# batches instead of paying for the whole train pool saves real wall-clock
# time on calls this expensive, not a rounding error on an already-cheap one.
# Rough budget math at --n-trials 300 (this script's default, matching
# optuna_search_pruning_warming.slurm's cap) even assuming ZERO pruning
# benefit (worst case, every trial runs to completion unpruned):
#   300 trials x 60 series x ~15s avg / 24 cores ~= 3.1h
# Any actual pruning only brings that down further. So the existing 2-day
# walltime below is generous headroom even in the pessimistic case, not a
# tight budget -- kept unchanged from the ruptures SLURM scripts for
# consistency. 24 cpus / 175G mem are likewise unchanged: KS_test has no
# analog of Dynp's memory blowups (see the ruptures pruning SLURM script's own
# history comment, not relevant here), so there's no reason to expect this
# run to need more or less memory.
#
# --n-trials is a hard cap on TOTAL trials across every resumed invocation.
# --p-threshold 0.05 is a deliberately conservative first guess (stricter than
# Optuna's 0.1 default), matching optuna_search_pruning_warming.slurm's own
# choice, given this dataset's series vary in noise/transition character; if
# validation score gaps vs. the patience run's are large, lower it further
# next time.
#SBATCH --job-name=optuna_search_kstest_pruning_warming
#SBATCH --account=julieneg_1707
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=175G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/optuna_search_kstest_pruning_warming_%j.out
#SBATCH --error=logs/optuna_search_kstest_pruning_warming_%j.err
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

$PYTHON -c "
import optuna
from optuna.pruners import WilcoxonPruner
WilcoxonPruner(p_threshold=0.05, n_startup_steps=1)
print(f'optuna {optuna.__version__} with WilcoxonPruner(n_startup_steps=...) OK')
" || {
    echo "ERROR: optuna.pruners.WilcoxonPruner is missing or doesn't accept n_startup_steps -- "
    echo "check/upgrade the optuna version in ammonyte_env (pip install -U optuna)"
    exit 1
}

# Prevent thread oversubscription (parallelism comes from joblib processes)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd ~/Syntheticdata/optimization_warming

$PYTHON -u optuna_search_kstest_pruning.py \
    --objective warming \
    --train-path ~/Syntheticdata/train_test_split_data/train.npz \
    --val-path ~/Syntheticdata/train_test_split_data/validation.npz \
    --outdir results_ks_pruning \
    --study-name search_kstest_warming_pruning \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --n-trials 300 \
    --p-threshold 0.05 \
    --max-call-seconds 7200 \
    --timeout 600000

echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
