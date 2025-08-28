#!/bin/bash
#SBATCH --job-name=dit_t2m
#SBATCH --time=0-00:20:00
#SBATCH --account=training2533
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
#SBATCH --reservation=mlesm_hackathon_2
#SBATCH --output=./logs/%x.%j.out
#SBATCH --error=./logs/%x.%j.err

ml GCCcore/13.3.0 Python/3.12.3-GCCcore-13.3.0

# environmental variables to support cpus_per_task with Slurm>22.05
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

#activate virtual environment
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3,5,6,7"
export NCCL_SOCKET_IFNAME=ib0
export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr}

echo "Start time: $(date +%T)"
srun --overlap python train.py --cur_epoch 240 --resume_from_ckpt  /p/project1/training2533/patnala1/WeGenDiffusion/results/DiT-B-2/ckpt_0000240.pt --global-batch-size 4 --epochs 300
echo "End time: $(date +%T)"

