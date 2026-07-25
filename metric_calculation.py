import sast.env  # noqa: F401  isort:skip -- must precede torch/numba imports

import pandas as pd
import numpy as np
import torch
import torch.utils.data as torch_data
from pathlib import Path
import logging
import scipy

from bam_poses.eval import EvalNDMS, Evaluation, load_results, save_results
from sast.realism_classifier.model import RealismClassifier
from sast.realism_classifier.data import process_eval_pkl
from sast.metrics import calculate_ndms_k, calculate_means, unique_ratios
from sast.utils import startup

data_dir = "output/"

# Root of the BAM-poses dataset, same as cfg.data.bam_location. Needed to build
# the NDMS motion-word databases.
bam_data_path = "data/dataset/"

tmp_dir = "tmp_sast/"

device = "cuda"

# eval.py output (or the equivalent for a baseline), and the number of observed
# frames that model was given. n_in matters for the NDMS metrics, which have to
# skip the motion words that still overlap the observed sequence.
eval_files = {
    "SiMLPe": ("mlp_D.pkl", 50),
    "MRT": ("mrt_D.pkl", 25),
    "Ours": ("Ours_D.pkl", 25),
    "TriPod": ("tripod_D.pkl", 25),
    "HisRep": ("hisrep_D.pkl", 50),
}

## Realism Scores

model = RealismClassifier.load("realism_classifier.pth", device)


def calc_per_second(path, gt=False):
    seqs = process_eval_pkl(path, gt=gt)

    # Score the 84% test split only: the classifier was trained on the
    # complementary 16% of the sequences of the models it was fitted against, so
    # scoring everything would report partly on its own training data. Same
    # generator seed and proportions as
    # sast.realism_classifier.data.synthetic_whole_sequences, which is what
    # produced the *_test_seq.pkl files this used to read. Applied to every
    # model so they are all scored on the same subset.
    gen = torch.Generator().manual_seed(42)
    _, test = torch_data.random_split(seqs, [0.16, 0.84], generator=gen)

    preds = model.forward_seq(test, device=device)
    results = np.stack([t.numpy() for t in preds])

    means = {}

    for i in range(9):
        # aggregate scores over all short sequences for first 2+i seconds
        means[i+2] = np.nanmean(results[:, :1+i*5])

    return results, means

means = {}

for name, (file, _) in eval_files.items():
    raw, m = calc_per_second(data_dir + file)
    means[name] = m

# ground truth motion, for reference. Any eval file will do, they all carry the
# same ground-truth futures.
_, means["GT"] = calc_per_second(data_dir + eval_files["Ours"][0], gt=True)

means_df = pd.DataFrame(means)
print(means_df)

## NDMS-based metrics

# Apply NDMS calculation
def eval_ndms(ndms, results_path):

    results_path = Path(results_path)

    results = load_results(results_path)

    logging.info("Running NDMS for %s", results_path.name)

    avg_ndms, avg_indices = ndms.run(results)

    logging.info("Saving results")

    save_path = results_path.parent / f"{results_path.stem}_ndms.pkl"
    save_path_idxs = results_path.parent / f"{results_path.stem}_ndms_indices.pkl"

    save_results(save_path, avg_ndms)
    save_results(save_path_idxs, avg_indices)

    logging.info("done")

startup(no_config=True)

ev = Evaluation(
    dataset="D",
    data_location=bam_data_path,
    tmp_dir=tmp_dir,
    n_in=25,
    n_out=250,
)
ndms = EvalNDMS(ev)

for name, (file, _) in eval_files.items():
    eval_ndms(ndms, data_dir + file)


## Aggregate NDMS, calculate NDMS@k

# eval_ndms() wrote these next to their input, as <stem>_ndms.pkl
ndms_stems = {
    name: (data_dir + Path(file).stem, n_in)
    for name, (file, n_in) in eval_files.items()
}

results_ndms = {k : calculate_means(load_results(v + "_ndms.pkl"), n_in) for k, (v, n_in) in ndms_stems.items()}

ndmsk_df = calculate_ndms_k(results_ndms)

print(ndmsk_df)

## UMWR

umwr_df = pd.DataFrame({k: unique_ratios(load_results(v + "_ndms_indices.pkl"), n_in) for k, (v, n_in) in ndms_stems.items()})

print(umwr_df)

## Trajectory distances

def total_distance_dist(eval, gt=False):
    abs_distances = []
    cum_distances = []
    for cat, data in eval.items():
        for sample in data:
            sample = sample["seq_out_pred"] if not gt else sample["seq_out_gt"]
            root_traj = (sample[..., 13, :2] + sample[..., 14, :2]) / 2

            if len(root_traj.shape) == 4:
                assert root_traj.shape[0] == 1, str(root_traj.shape)
                root_traj = root_traj[0]

            # root_traj: (p t d)

            abs_d = np.linalg.norm(root_traj[:, 0] - root_traj[:, -1], axis=-1)
            cum_d = np.sum(np.linalg.norm(np.diff(root_traj, axis=1), axis=-1), axis=1)

            abs_distances.append(abs_d)
            cum_distances.append(cum_d)

    a = np.concatenate(abs_distances, axis=0)
    c = np.concatenate(cum_distances, axis=0)

    return a, c

abs_distances = {}
cum_distances = {}

for name, (file, _) in eval_files.items():
    abs_distances[name], cum_distances[name] = total_distance_dist(
        load_results(data_dir + file)
    )

# the ground-truth futures are the same in every eval file
abs_distances["GT"], cum_distances["GT"] = total_distance_dist(
    load_results(data_dir + eval_files["Ours"][0]), gt=True
)

values = {
    k : {
        'mean': v.mean(),
        'median': np.median(v),
        'std': v.std(),
        'w1': scipy.stats.wasserstein_distance(v, cum_distances['GT'])
    }
    for k, v in cum_distances.items()
}

print(pd.DataFrame(values).T.sort_values('median'))