#!/bin/bash
#SBATCH --job-name=dit_t2m_test
#SBATCH --time=01:00:00
#SBATCH --account=training2533
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=dc-gpu
#SBATCH --output=./logs/%x.%j.out
#SBATCH --error=./logs/%x.%j.err
#SBATCH --reservation=mlesm_hackathon_3

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
#torchrun --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT sample_ddp.py --ckpt /fast/project/HFMI_HClimRep/nishant.kumar/dit_hackathon/results/DiT-XL-2/ckpt_0032000.pt --output-dir ./samples 

#srun --overlap python sample_ddp.py --ckpt results-months/ckpt_0000100.pt --output-dir ./samples-months --test_type month --label 0
#srun --overlap python sample_ddp.py --ckpt results-months/ckpt_0000100.pt --output-dir ./samples-months --test_type month --label 3
#srun --overlap python sample_ddp.py --ckpt results-months/ckpt_0000100.pt --output-dir ./samples-months --test_type month --label 6
#srun --overlap python sample_ddp.py --ckpt results-months/ckpt_0000100.pt --output-dir ./samples-months --test_type month --label 9


#srun --overlap python sample_ddp.py --ckpt /p/project1/training2533/lancelin1/WeGenDiffusion/results/DiT-B-2_season_1/ckpt_0000110.pt --output-dir ./samples-season --test_type season --label 0
#srun --overlap python sample_ddp.py --ckpt /p/project1/training2533/lancelin1/WeGenDiffusion/results/DiT-B-2_season_1/ckpt_0000110.pt --output-dir ./samples-season --test_type season --label 1
#srun --overlap python sample_ddp.py --ckpt /p/project1/training2533/lancelin1/WeGenDiffusion/results/DiT-B-2_season_1/ckpt_0000110.pt --output-dir ./samples-season --test_type season --label 2
#srun --overlap python sample_ddp.py --ckpt /p/project1/training2533/lancelin1/WeGenDiffusion/results/DiT-B-2_season_1/ckpt_0000110.pt --output-dir ./samples-season --test_type season --label 3

#srun --overlap python sample_ddp.py --ckpt /p/project1/training2533/lancelin1/WeGenDiffusion/results/DiT-B-2_previous_state_1/ckpt_0000110.pt --output-dir ./samples-previous --test_type previous_state

srun --overlap python sample_ddp.py --ckpt /p/project1/training2533/corradini1/WeGenDiffusion/results/DiT-B-2_test/ckpt_0000240.pt --output-dir ./samples 
