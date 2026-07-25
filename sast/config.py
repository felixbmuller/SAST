from yacs.config import CfgNode
import pytorch_lightning as pl

cfg = CfgNode()

#######################
# Experiment-specific #
#######################
cfg.experiment = CfgNode()

# Run-name is set to the name of the config file
cfg.experiment.run_name = ""  # set automatically
cfg.experiment.git_revision_hash = ""  # set automatically
cfg.experiment.viz_every_n_steps = 20
cfg.experiment.chkpt_every_n_epochs = 20
cfg.experiment.chkpt_every_n_steps = 20000
cfg.experiment.resume_from_chkpt = None
cfg.experiment.comment = ""
cfg.experiment.study_name = "default"


##############################################################
# Meta-information about data that is needed by some modules #
##############################################################

cfg.data = CfgNode()

# constants
cfg.data.n_joints = 17
cfg.data.fps = 25

# settings for BampData
cfg.data.frames_in = 25
cfg.data.frames_out = 279

cfg.data.object_frame = 24

# stride of the sliding window used to cut sequences out of the recordings.
# Note that this is *smaller* than frames_in + frames_out, i.e. windows overlap
# and every frame ends up in (frames_in + frames_out) / seq_offset sequences.
# This drives the size of the generated dataset on disk, but not the memory
# needed to generate it (see splits_per_batch in create_dataset for that).
cfg.data.seq_offset = 50

# number of different random object point-cloud samplings stored per person.
# One of them is drawn at random in MultiPersonData.__getitem__.
cfg.data.object_embeds = 8

cfg.data.n_chunks = 10

# root of the BAM-poses dataset, i.e. the directory that contains one
# subdirectory per recording (A/, B/, C/, D/) plus the scene geometry
# (A_scene/, ...). See the README on how to obtain it.
cfg.data.bam_location = "data/dataset/"


###############
# Data Loader #
###############

cfg.loader = CfgNode()

cfg.loader.dataset_path = "data/"

cfg.loader.dataset = "bamp"

cfg.loader.batch_size = 32

cfg.loader.shuffle_train = True
cfg.loader.num_workers = 6

cfg.loader.data_mask = "half_only_standing_mask"
cfg.loader.dataset_parts = ["A", "B", "C"]

########
# UNet #
########

cfg.unet = CfgNode()

cfg.unet.time_embed_dim = 64

cfg.unet.primary_tcn = CfgNode()

# the input sequence is concatenated onto the noisy sample along the joint
# axis, hence the factor 2 (see MultiTcnDiffusion.forward)
cfg.unet.primary_tcn.channels = [cfg.data.n_joints * 3 * 2, 128, 256, 512]
cfg.unet.primary_tcn.kernel_size = 5
cfg.unet.primary_tcn.dropout = 0.2
cfg.unet.primary_tcn.norm_mode = "group_norm"  # options: none, group_norm
cfg.unet.primary_tcn.use_residual = True
cfg.unet.primary_tcn.group_norm_groups = 32
cfg.unet.primary_tcn.timestep_mode = "add"
cfg.unet.primary_tcn.padding_mode = "zero"


cfg.unet.others_tcn = CfgNode()

cfg.unet.others_tcn.channels = [cfg.data.n_joints * 3, 128, 128, 128]
cfg.unet.others_tcn.kernel_size = 3
cfg.unet.others_tcn.dropout = 0.2
cfg.unet.others_tcn.norm_mode = "group_norm"  # options: none, group_norm
cfg.unet.others_tcn.use_residual = True
cfg.unet.others_tcn.group_norm_groups = 32
cfg.unet.others_tcn.timestep_mode = "add"
cfg.unet.others_tcn.padding_mode = "zero"


cfg.unet.person_transformer = CfgNode()

cfg.unet.person_transformer.d_model = 128
cfg.unet.person_transformer.nhead = 4
cfg.unet.person_transformer.num_encoder_layers = 2
cfg.unet.person_transformer.num_decoder_layers = 2
cfg.unet.person_transformer.dropout = 0.1
cfg.unet.person_transformer.dim_feedforward = 512


cfg.unet.scene_transformer = CfgNode()

cfg.unet.scene_transformer.d_model = 256
cfg.unet.scene_transformer.nhead = 8
cfg.unet.scene_transformer.num_encoder_layers = 3
cfg.unet.scene_transformer.num_decoder_layers = 3
cfg.unet.scene_transformer.dropout = 0.1
cfg.unet.scene_transformer.dim_feedforward = 1024

cfg.unet.mask_memory_key = False  # Transformer setting
cfg.unet.causal = True

# For ablations
cfg.unet.remove_scene = False
cfg.unet.remove_others = False

cfg.unet.time_embedding_type = "fourier"

#############
# Diffusion #
#############

cfg.diffusion = CfgNode()

cfg.diffusion.num_train_timesteps = 1000
cfg.diffusion.beta_schedule = "squaredcos_cap_v2"
cfg.diffusion.clip_sample = False


#####################################
# Optimizer & LR scheduler settings #
#####################################

# only Adam at the moment
cfg.optim = CfgNode()

cfg.optim.lrate = 1e-4  # diffuser default 1e-4
cfg.optim.warmup_steps = 500 

cfg.optim.per_step = False

#####################
# Trainer Arguments #
#####################

cfg.ztrainer = CfgNode()


cfg.ztrainer.accelerator = "auto"
cfg.ztrainer.devices = 1
cfg.ztrainer.limit_val_batches = 0
cfg.ztrainer.max_steps = 1500000
cfg.ztrainer.gradient_clip_val = 1.0


def get_cfg_defaults():
    return cfg.clone()
