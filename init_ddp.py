import os
import torch
import torch.distributed as dist

def init_distributed_mode():
    world_size = int(os.environ['SLURM_NTASKS'])
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(local_rank)

