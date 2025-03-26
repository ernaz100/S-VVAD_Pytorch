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
SINGULARITY_IMAGE="/home/ponsmoll/pba794/S-VVAD_Pytorch/image.sif"

# Use SLURM's automatic temporary directory
SCRATCH_DIR="${TMPDIR}"

echo "Using temporary directory: ${SCRATCH_DIR}"

# Copy Singularity image to scratch with verification
echo "Copying Singularity image to scratch..."
cp -v "${SINGULARITY_IMAGE}" "${SCRATCH_DIR}/"

# Verify the copy was successful
if [ ! -f "${SCRATCH_DIR}/image.sif" ]; then
    echo "Error: Failed to copy Singularity image to scratch directory"
    exit 1
fi

# Load singularity module
module load singularity

# Run the Python script within Singularity container
singularity exec --nv \
    --contain \
    --bind "${SCRATCH_DIR}":/scratch \
    --bind /etc/passwd:/etc/passwd \
    --bind /etc/group:/etc/group \
    --bind /usr/lib64:/usr/lib64:ro \
    --bind /lib64:/lib64:ro \
    "${SCRATCH_DIR}/image.sif" \
    python3 -m src.train_resnet_vad

