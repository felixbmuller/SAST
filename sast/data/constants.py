objects_per_scene = {
    "A": 32,
    "B": 31,
    "C": 24,
    "D": 46,
}

max_objects_per_scene = 46

object_embed_dim = 2061

n_in = 25

max_persons_simultaneous = 16

normalize_frame = 24

# the six activity groups the BAM-poses benchmark evaluates on, see
# bam_poses.eval.evaluation.ACTIVITIES_FOR_EVAL
EVAL_CATEGORIES = [
    "COFFEE MACHINE",
    "WHITEBOARD",
    "FRIDGE",
    "OPEN DRAWERS AND CUPBOARDS",
    "USE SINK",
    "SITTING DOWN",
]
