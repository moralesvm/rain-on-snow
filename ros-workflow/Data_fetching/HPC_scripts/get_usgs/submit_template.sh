#!/bin/sh

# Run on the short partition because these are quick jobs
#SBATCH --partition=general

#SBATCH --account=account

# Request one node
#SBATCH --nodes=1

# Request one task
#SBATCH --ntasks=1

# Request 6 CPUs per task
#SBATCH --cpus-per-task=4

# Request 24GB of RAM
#SBATCH --mem=72G

# Run for a maximum of 72 hours
#SBATCH --time=30:00:00

# Name of the job
#SBATCH --job-name=USGS$YEAR

# Name the output file
#SBATCH --output=USGS$YEAR.out

# Set email address for notifications
#SBATCH --mail-user=your_email@.com

# Request email to be sent at both begin and end, and if job fails
#SBATCH --mail-type=ALL

####  End Slurm commands

## Setup the conda/python environment
module purge
module load miniforge/25.11.0-py3.12

conda activate rosEnvUpdt
echo "Displaying conda environments:  rosEnv should have * when active"
conda env list
which python
python --version

# Force the allocator to return freed memory to the OS instead of letting
# "unmanaged memory" pile up toward the per-worker cap (the proximate cause of
# the intermittent KilledWorker failures). Must be set before numpy/dask import.
export MALLOC_TRIM_THRESHOLD_=0

## Use the -u flag to make output print as soon as it can
python -u path_to_script/get_Q_usgs_wArgs.py --year $YEAR
