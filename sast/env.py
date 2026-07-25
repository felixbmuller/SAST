"""
Process-wide environment defaults.

Import this *first* in every entry point, before torch, numba or anything that
pulls them in -- the variables set here are only read once, when those libraries
are imported, so setting them later has no effect.
"""

import os


def setup():
    # numba (bam_poses) and torch each link their own OpenMP runtime. Letting
    # both spin up a full thread pool oversubscribes the machine, and on some
    # platforms the two runtimes in one process crash outright (observed on
    # macOS/arm64 with conda numba + pip torch: a segfault as soon as a torch
    # model is built in a process that has run a numba kernel).
    #
    # Preprocessing and the dataloader get their parallelism from worker
    # *processes* (cfg.loader.num_workers), not from OpenMP threads, so one
    # OpenMP thread per process is the right default here.
    #
    # Override from the shell if you want the threads back, e.g.
    #   OMP_NUM_THREADS=8 python train.py SAST.yaml
    os.environ.setdefault("OMP_NUM_THREADS", "1")


setup()
