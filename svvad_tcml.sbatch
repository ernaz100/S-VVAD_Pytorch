#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=S-VVAD

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:4
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=23:59:59
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"


#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err
#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=eric.nazarenus@student.uni-tuebingen.de
# your mail address

singularity exec --nv /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg python -m src.train_resnet_vad

echo DONE!

