#!/bin/bash
#SBATCH --job-name=dit_t2m_test
#SBATCH --time=1-00:00:00
#SBATCH --account=hclimrep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --output=./logs/%x.%j.out
#SBATCH --error=./logs/%x.%j.err

#import modules and activate the virtual
ml GCCcore/13.3.0 Python/3.12.3-GCCcore-13.3.0

# environmental variables to support cpus_per_task with Slurm>22.05
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

#activate virtual environment
#source /fast/home/nishant.kumar/venv_python_3.10/bin/activate
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_SOCKET_IFNAME=ib0
export MASTER_PORT=12341
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr}

#run the python file
#torchrun --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT sample_ddp.py --ckpt /fast/project/HFMI_HClimRep/nishant.kumar/dit_hackathon/results/DiT-XL-2/ckpt_0032000.pt
srun --overlap python sample_ddp.py --ckpt results/DiT-XL-2/ckpt_0099500.pt --output-dir ./samples
