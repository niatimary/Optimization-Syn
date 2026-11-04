#!/bin/bash
# Submit from ~/Syntheticdata/optimization_warming
# (`cd ~/Syntheticdata/optimization_warming && sbatch optuna_search_kstest_warming.slurm`).
# results_ks/ must exist before submission -- SLURM opens --output/--error at job
# start, before this script's own commands run, and this script does not
# create it for you. (No logs/ dir here -- --output/--error point straight at
# results_ks/, unlike the ruptures scripts' logs/ convention.) Named results_ks/,
# NOT results/ -- this directory also holds optuna_search.py/optuna_search_pruning.py
# (the ruptures scripts), which already own results/ and results_pruning/; reusing
# either name here would have this script's optuna_trials.csv/best_params.json/
# study.db/etc overwrite theirs, since all these scripts are submitted from this
# same optimization_warming/ directory.
#
# Layout expected:
#   ~/Syntheticdata/train_test_split_data/{train,validation,test}.npz  (data)
#   ~/Syntheticdata/optimization_warming/{optuna_search_kstest.py,
#       optuna_search_kstest_warming.slurm, results_ks/}
#
# KS test (ammonyte.Series.kstest / KS_test, Bagniewski et al. 2021) is
# roughly 500-1000x more expensive per call than the ruptures methods this
# project's other Optuna scripts tune: measured 12-22s per KS_test call on
# this project's actual train.npz (60 series, 6001 points each), versus
# ruptures' ~0.02-0.03s. Rough budget math at --n-trials 250 (this script's
# default -- raised above ruptures' 130/50/30 patience/min-trials/patience
# numbers deliberately: unlike ruptures' method/cost branching, every KS-test
# trial refines the same flat 6-parameter space, so there's no branch-
# exploration cost traded away by letting the study run longer here):
#   250 trials x 60 series x ~15s avg / 24 cores ~= 2.6h
# So the existing 2-day walltime below is generous headroom, not a tight
# budget -- kept unchanged from the ruptures SLURM scripts for consistency,
# not because this run is expected to need anywhere near it. 24 cpus / 175G
# mem are likewise unchanged from the ruptures scripts: KS_test has no
# analog of Dynp's memory blowups (see optuna_search_warming.slurm's own
# history comment, not relevant here), so there's no reason to expect this
# run to need more or less memory than the ruptures runs did.
#
# --n-trials is a hard cap on TOTAL trials across every resumed invocation --
# the study stops once it's reached, possibly well before --time is up.
# --min-trials/--patience/--min-delta are the study-level early-stopping rule
# (see optuna_search_kstest.py's module docstring) -- this is the DELIBERATE
# point of comparison against optuna_search_kstest_pruning.py's WilcoxonPruner
# per-trial pruning, run as a separate study (in optuna_search_kstest_pruning/)
# -- compare both best_params.json on validation once both have run. If the
# --n-trials cap isn't reached in one submission for any reason, just resubmit
# this same script -- the study is checkpointed (--storage) and resumes from
# wherever it left off, no trials lost.
#SBATCH --job-name=optuna_search_kstest_warming
#SBATCH --account=julieneg_1707
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=175G
#SBATCH --time=2-00:00:00
#SBATCH --output=results_ks/optuna_search_kstest_warming_%j.out
#SBATCH --error=results_ks/optuna_search_kstest_warming_%j.err
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

$PYTHON -u optuna_search_kstest.py \
    --objective warming \
    --train-path ~/Syntheticdata/train_test_split_data/train.npz \
    --val-path ~/Syntheticdata/train_test_split_data/validation.npz \
    --outdir results_ks \
    --study-name search_kstest_warming \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --n-trials 250 \
    --min-trials 100 \
    --patience 50 \
    --min-delta 0.01 \
    --timeout 165000

echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
