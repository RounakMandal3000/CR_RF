#!/bin/bash
#SBATCH --job-name="cloud_vae_v3"              # Name of the job
#SBATCH -p gpu                                    # Name of the partition: available options "standard, standard-low, gpu, hm"
#SBATCH --error=cloud_vae_v3.err                 # Name of stderr error file
#SBATCH --output=cloud_vae_v3.out                # Name of stdout output file
#SBATCH -N 1                                      # Number of nodes
#SBATCH -n 1                                      # Number of processes
#SBATCH -c 8                                      # Number of CPU cores per process
#SBATCH --gres=gpu:1                              # Request 2 GPU in original
#SBATCH -t 70:00:00                               # Walltime in HH:MM:SS (max 72:00:00)
#SBATCH --mail-user=rounakmandal3000@gmail.com       # Email ID for job status notifications
#SBATCH --mail-type=ALL                           # Send mail for all types of job events
#SBATCH --mem=32G

#source /home/21ec39053/anaconda3/bin/activate BTP_env_v2
source /home/21ec39053/miniconda3/etc/profile.d/conda.sh
conda activate cloud

# Path to the Python script you want to run
exe="/scratch/21ec39053/MTP/vae_cloudless_training.py"                       # Change to the path of your Python script

python "$exe"

#torchrun --standalone --nproc_per_node=2 "$exe"