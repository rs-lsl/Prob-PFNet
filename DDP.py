# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from tqdm import tqdm
import sys
import tempfile
import time

import torch.distributed as dist
from torch.utils.data import DataLoader

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # print("NOT using distributed mode")
        raise EnvironmentError("NOT using distributed mode")
        # return
    # print(args)
    #
    args.distributed = True

    # Need to set the GPU to use here.
    torch.cuda.set_device(args.gpu)
    # This is the communication method between GPUs. There are several options; nccl is faster and recommended.
    args.dis_backend = 'nccl'
    # Initialize multi-GPU.
    dist.init_process_group(
        backend=args.dis_backend,
        init_method=args.dis_url,
        world_size=args.world_size,
        # timeout=timedelta(seconds=7200000),
        rank=args.rank,
        device_id=torch.device(f"cuda:{args.gpu}")
    )
    # This synchronizes across GPUs: some GPUs may run faster and some slower
    # (for example, if you check if RANK == 0: do something, then process 0 will execute extra code and become slower).
    # So this code waits for all processes to reach this point.
    dist.barrier()


def init_distributed_mode_old(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # print("NOT using distributed mode")
        raise EnvironmentError("NOT using distributed mode")
        # return
    # print(args)
    #
    args.distributed = True

    # Need to set the GPU to use here.
    torch.cuda.set_device(args.gpu)
    # This is the communication method between GPUs. There are several options; nccl is faster and recommended.
    args.dis_backend = 'nccl'
    # Initialize multi-GPU.
    dist.init_process_group(
        backend=args.dis_backend,
        init_method=args.dis_url,
        world_size=args.world_size,
        # timeout=timedelta(seconds=7200000),
        rank=args.rank
    )
    # This synchronizes across GPUs: some GPUs may run faster and some slower
    # (for example, if you check if RANK == 0: do something, then process 0 will execute extra code and become slower).
    # So this code waits for all processes to reach this point.
    dist.barrier()


def cleanup():
    # No need to say much here; the name makes it clear.
    dist.destroy_process_group()

# Check whether multi-GPU is available and initialized.
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# Get the number of GPUs/processes. Mainly used for all_reduce computation.
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# Get the rank of the process.
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# This is mainly similar to all_reduce: each process computes a value, and the values from different processes are combined.
# For example, for loss, process 0 gets loss 0.1 for samples 1 and 2, and process 1 gets loss 0.2 for samples 3 and 4, then
# the all_loss for this batch is 0.1 + 0.2 = 0.3
# Similarly, when computing classification accuracy, for example, if the batch size is 100, process 0 gets 50 correct
# and process 1 gets 60 correct, then the overall accuracy is 110/200
def reduce_value(value, average=True):
    # Get the number of GPUs, mainly to determine how many processes we have.
    world_size = get_world_size()
    # If there is only one process, return.
    if world_size < 2:
        return value

    with torch.no_grad():
        # This is all_reduce, which aggregates and returns the values from different processes.
        dist.all_reduce(value)
        if average:
            # Whether to average.
            value /= world_size
        return value

# Check whether this is the main process. The main process means rank=0.
# Strictly speaking, there is no distinction of a main process; if you want process 1 to be the main process,
# just use get_rank() == 1.
def is_main_process():
    return get_rank() == 0

def clip_grads(params, args, norm_type: float = 2.0):
    """ Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    args.clip_mode = args.clip_mode if args.clip_grad is not None else None
    if args.clip_mode is None:
        return
    if args.clip_mode == 'norm':
        torch.nn.utils.clip_grad_norm_(params, args.clip_grad, norm_type=norm_type)
    elif args.clip_mode == 'value':
        torch.nn.utils.clip_grad_value_(params, args.clip_grad)
    else:
        assert False, f"Unknown clip mode ({args.clip_mode})."


if __name__ == '__main__':
    pass