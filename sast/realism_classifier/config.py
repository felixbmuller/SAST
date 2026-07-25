from pathlib import Path
from yacs.config import CfgNode

cfg = CfgNode()

cfg.n_joints = 29
cfg.n_frames = 50
cfg.n_hidden_temporal = 32
cfg.n_hidden_final = 512

cfg.version = 1

cfg.lrate = 1e-3

cfg.device = "cuda"
cfg.data_path = str(Path(__file__).parent.parent.parent.parent / "data/realism_classifier")
cfg.train_batch_size = 16
cfg.val_batch_size = 4096
cfg.n_epochs = 50


def get_cfg_defaults():
    return cfg.clone()
