#!/bin/bash   
#SBATCH --mem=64G         
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-user=s.sourbron@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=kssa
#SBATCH --output=logs/kssa.out
#SBATCH --error=logs/kssa.err

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can 
# prevent performance issues if your code manages threading itself 
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND

# Ensures that all your environment variables from the submission 
# environment are passed into the job’s environment
export SLURM_EXPORT_ENV=ALL

# Loads the Anaconda module provided by the cluster.
# (On HPC systems, software is usually installed as “modules” to avoid version conflicts.)
module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC

# Define path variables here
ENV="/mnt/parscratch/users/$(whoami)/envs/template"
CODE="/mnt/parscratch/users/$(whoami)/scripts/ppln-template/src/template"
BUILD="/mnt/parscratch/users/$(whoami)/data/demo"
ARCHIVE="login1:/shared/abdominal_imaging/Archive/demo"

# srun runs your program on the allocated compute resources managed by Slurm
srun "$ENV/bin/python" "$CODE/ppln.py" --build="$BUILD"
rsync -av --no-group --no-perms "$BUILD/template" "$ARCHIVE"