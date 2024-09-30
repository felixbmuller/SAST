import pathlib
import subprocess
import logging
from typing import List
import sys
import random
from datetime import datetime

import numpy as np
import torch


BASE_PATH = pathlib.Path(__file__).parent.parent.resolve()


def startup(
    cfg_path=None,
    fix_seed=False,
    seed=4224160221,
    no_config=False,
):

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if fix_seed:
        set_seed(seed)

    if not no_config:
        return load_config(cfg_path)
    else:
        return


def count_parameters(m: torch.nn.Module, only_trainable: bool = True):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def timestamp():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S%Z")


def set_seed(seed: int, fully_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(cfg_path=None):
    """Load config from config file given as cli parameter and set run_name accordingly

    Returns
    -------
    CfgNode
        config
    """
    from sast.config import get_cfg_defaults

    if cfg_path is None:

        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} <config file>")
            print("See config.py for available options.")
            exit(0)

        cfg_path = str((BASE_PATH / sys.argv[1]).resolve())

    cfg_name = cfg_path.split("/")[-1]
    cfg_name = cfg_name.split(".")[0]

    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)

    cfg.experiment.run_name = cfg_name
    cfg.experiment.git_revision_hash = get_git_revision_hash()

    if cfg.experiment.resume_from_chkpt:
        cfg.experiment.resume_from_chkpt = str(
            (BASE_PATH / cfg.experiment.resume_from_chkpt).resolve()
        )
    else:
        cfg.experiment.resume_from_chkpt = None

    cfg.freeze()

    return cfg


def log_exception(x):
    logging.error(f"EXCEPTION: {type(x)} {x}")


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
