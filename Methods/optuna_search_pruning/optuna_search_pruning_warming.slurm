#!/bin/bash
# Submit from ~/Syntheticdata/optimization_warming
# (`cd ~/Syntheticdata/optimization_warming && sbatch optuna_search_pruning_warming.slurm`).
# logs/ and results_pruning/ already exist from the earlier (Dynp-inclusive) run --
# nothing new to create before submission.
#
# optuna_search_pruning.py no longer includes Dynp as a method (see its module docstring
# for why). This changes the search space, so before resubmitting under the SAME
# --study-name/--outdir as the earlier (Dynp-inclusive, scancel'd) run, the old study's
# entries in results_pruning/study.db must be cleared first -- otherwise the script's own
# resume-compatibility check (_search_space_fingerprint) correctly refuses to run at all.
# One-time cleanup (run once on CARC, with ammonyte_env active, from results_pruning/):
#   python -c "import optuna; optuna.delete_study(study_name='search_warming_pruning', storage='sqlite:///study.db')"
# That run was scancel'd mid-search anyway -- it never reached the code that writes
# optuna_trials.csv/best_params.json/convergence files, so nothing else is lost.
#
# WilcoxonPruner variant of optuna_search_warming.slurm (patience-based early
# stopping, untouched) -- runs optuna_search_pruning.py with its own --outdir/
# --study-name so results, logs, and the Optuna study never mix with the
# patience run's. Compare results_pruning/best_params.json against
# results/best_params.json on validation before trusting either for TEST.
#
# 175G mem, oneweek partition: back down from 500G/largemem now that
# optuna_search_pruning.py enforces --max-call-seconds (default 7200s = 2h hard
# cap on a single series' _detect call -- see DEFAULT_MAX_CALL_SECONDS). The
# earlier 500G/largemem move was a workaround for every Dynp trial in the
# PATIENCE run (optuna_search.py, no such cap) dying with SIGKILL at
# 175G/--n-jobs=24 regardless of n_bkps -- i.e. those trials were running
# unbounded until memory blew the ceiling. The real fix is capping wall-time
# per call (an unacceptably slow/hung config now fails that one trial cleanly,
# see CallTimeoutError), not raising the memory ceiling to let it keep going --
# so 175G/oneweek is expected to be enough again.
#
# --n-trials is a hard cap on TOTAL trials across every resumed invocation.
# --p-threshold 0.05 is a deliberately conservative first guess (stricter than
# Optuna's 0.1 default) given this dataset's series vary in noise/transition
# character; if validation score gaps vs. the patience run's are large,
# lower it further next time.
#SBATCH --job-name=optuna_search_pruning_warming
#SBATCH --account=julieneg_1707
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=175G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/optuna_search_pruning_warming_%j.out
#SBATCH --error=logs/optuna_search_pruning_warming_%j.err
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

$PYTHON -u optuna_search_pruning.py \
    --objective warming \
    --train-path ~/Syntheticdata/train_test_split_data/train.npz \
    --val-path ~/Syntheticdata/train_test_split_data/validation.npz \
    --outdir results_pruning \
    --study-name search_warming_pruning \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --n-trials 300 \
    --p-threshold 0.05 \
    --max-call-seconds 7200 \
    --timeout 600000

echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
