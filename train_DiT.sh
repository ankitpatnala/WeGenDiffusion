#!/bin/bash -x
#SBATCH --job-name=fancy
#SBATCH --account=training2533
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --partition=standard
#SBATCH --reservation=mlesm_hackathon_2

ml GCCcore/13.3.0 Python/3.12.3-GCCcore-13.3.0

# environmental variables to support cpus_per_task with Slurm>22.05
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

#activate virtual environment
source /fast/home/nishant.kumar/venv_python_3.10/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3,5,6,7"
export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr}

torchrun train.py --global-batch-size 4
