#!/bin/bash
# Submit from ~/Syntheticdata (`sbatch methods/run_grid_experiments.slurm`).
# methods/logs/ must exist before submission -- SLURM opens --output/--error
# at job start, before this script's own commands run (`mkdir -p methods/logs`
# once from the login node is enough; a .gitkeep in that folder keeps it
# around across git pulls).
#SBATCH --job-name=grid_calibrated_snr3
#SBATCH --account=julieneg_1707
#SBATCH --partition=oneweek
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=methods/logs/grid_calibrated_snr3_%A_%a.out
#SBATCH --error=methods/logs/grid_calibrated_snr3_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niati@usc.edu

# One array task per ruptures method (4 methods, 0-3), all on the same
# ngrip-calibrated red/snr3 condition. Window was dropped (underperformed
# every other method across every cost tested). Dynp (dynamic programming,
# heavy) and the KS test are intentionally not included -- see
# run_grid_experiment.py.
METHODS=(binseg pelt bottomup kernelcpd)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID  Method: $METHOD"
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

cd ~/Syntheticdata
mkdir -p methods/logs

$PYTHON -u methods/run_grid_experiment.py --method "$METHOD" --n-jobs "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "Job completed ($METHOD): $(date)"
echo "========================================"
