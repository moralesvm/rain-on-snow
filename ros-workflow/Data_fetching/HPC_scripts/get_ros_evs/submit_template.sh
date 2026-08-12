#!/bin/sh

# Run on the short partition because these are quick jobs
#SBATCH --partition=general

# Request one node
#SBATCH --nodes=1

# Request one task
#SBATCH --ntasks=1

# Request 6 CPUs per task
#SBATCH --cpus-per-task=16

# Request 24GB of RAM
#SBATCH --mem=96G

# Run for a maximum of 72 hours
#SBATCH --time=30:00:00

# Name of the job
#SBATCH --job-name=Ros$HUC2_LIST

# Name the output file
#SBATCH --output=Ros$HUC2_LIST.out

# Set email address for notifications
#SBATCH --mail-user=your_email@.com

# Request email to be sent at both begin and end, and if job fails
#SBATCH --mail-type=ALL

####  End Slurm commands

## Setup the conda/python environment
#module load miniforge

#conda activate teehrClstr
# Initialize Conda (using the ANACONDA_ROOT variable)
#source ~/.bashrc
module purge
module load miniforge/25.11.0-py3.12

conda activate rosEnv
echo "Displaying conda environments:  rosEnv should have * when active"
conda env list
which python
python --version

## Use the -u flag to make output print as soon as it can.
python -u path_to_script/get_percRos_evs_perHuc2.py --HUC2_LIST $HUC2_LIST
