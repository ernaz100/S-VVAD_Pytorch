#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J svvad                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-fat-galvani   # Which partition will run your job
#SBATCH --time=0-23:59             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/home/ponsmoll/pba794/S-VVAD_Pytorch/myjob-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/home/ponsmoll/pba794/S-VVAD_Pytorch/myjob-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=eric.nazarenus@student.uni-tuebingen.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

# Setup Phase
# Copy Singularity image to local scratch space for better performance
SINGULARITY_IMAGE="/path/to/your/image.sif"
SCRATCH_DIR="/scratch_local/${SLURM_JOB_ID}"

# Create scratch directory
mkdir -p ${SCRATCH_DIR}

# Copy Singularity image to scratch
cp ${SINGULARITY_IMAGE} ${SCRATCH_DIR}/

# Run the Python script within Singularity container
singularity exec --nv \
    --bind ${SCRATCH_DIR}:/scratch \
    ${SCRATCH_DIR}/image.sif \
    python3 -m src.train_resnet_vad

# Cleanup scratch directory after job completion
rm -rf ${SCRATCH_DIR}