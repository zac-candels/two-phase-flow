#!/bin/bash
#SBATCH --job-name=CA_fn
#SBATCH --output=CA_fn_out.log
#SBATCH --error=CA_fn_err.log
#SBATCH --time=24:00:00

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1   # MPI ranks per node (adjust)
#SBATCH --cpus-per-task=1

#SBATCH --partition=standard
#SBATCH --account=ec288-wetting
#SBATCH --qos=lowpriority
#SBATCH --exclusive

# Load Cray Python (required for srun/MPI environment)
module load cray-python

# DO NOT load singularity unless module spider confirms it exists
# If singularity is installed system-wide, this is enough:
which singularity || which apptainers

# Pass cpus-per-task to srun (per HPC docs)
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

SIF=../../fenics-gmsh.sif
WORKDIR=$SLURM_SUBMIT_DIR

# Launch MPI ranks with srun
srun --distribution=block:block --hint=nomultithread \
     singularity exec \
     --bind $WORKDIR:/root/shared \
     $SIF \
     python3 /root/shared/run_test.py