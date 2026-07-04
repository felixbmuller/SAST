
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
import pickle
import scipy

from hik.eval import EvalNDMS, Evaluation, load_results, save_results
from sast.realism_classifier.model import RealismClassifier
from sast.metrics import calculate_ndms_k, calculate_means, unique_ratios
from sast.utils import startup

data_dir = "output/"

test_files = {
    "SiMLPe": "mlp_D_test_seq.pkl",
    "MRT": "mrt_D_test_seq.pkl",
    "Ours": "Ours_D_test_seq.pkl",
    "TriPod": "tripod_D_test_seq.pkl",
    "HisRep": "hisrep_D_test_seq.pkl",
    "GT": "gt_test_seq.pkl"
}

## Realism Scores

device = "cuda"
model = RealismClassifier.load("realism_classifier.pth", device)

def calc_per_second(path):
    data = torch.load(f"{data_dir}/{path}")
    preds = model.forward_seq(data, device=device)
    results = np.stack([t.numpy() for t in preds])

    means = {}

    for i in range(9):
        # aggregate scores over all short sequences for first 2+i seconds
        means[i+2] = np.nanmean(results[:, :1+i*5])

    return results, means

means = {}

for file in test_files.values():
    raw, m = calc_per_second(file)
    means[file.removesuffix("_test_seq.pkl")] = m

means_df = pd.DataFrame(means)
print(means_df)

## NDMS-based metrics

# Apply NDMS calculation
def eval_ndms(results_path, n_out=250, dataset="D"):

    startup()

    results_path = Path(results_path)

    results = load_results(results_path)

    logging.info("Creating evaluation object")

    ev = Evaluation(
        dataset=dataset,
        tmp_dir="tmp_sast/",
        n_in=25,
        n_out=n_out,
    )

    logging.info("Creating EvalNDMS object")

    ndms = EvalNDMS(ev)

    logging.info("Running NDMS")

    avg_ours, agv_indices = ndms.run(results)

    logging.info("Saving results")

    save_path = results_path.parent / f"{results_path.stem}_ndms.pkl"
    save_path_idxs = results_path.parent / f"{results_path.stem}_ndms_indices.pkl"

    save_results(save_path, avg_ours)
    save_results(save_path_idxs, agv_indices)

    logging.info("done")

for k, v in test_files.items():
    eval_ndms(data_dir + v)


eval_files = {
    "MRT": ("output/mrt_D_test_seq", 25),
    "HisRep": ("output/hisrep_D_test_seq", 50),
    "SiMLPe": ("output/mlp_D_test_seq", 50),
    "TriPod": ("output/tripod_D_test_seq", 25),
    "Ours": ("output/Ours_D_test_seq", 25),
}


## Aggregate NDMS, calculate NDMS@k

results_ndms = {k : calculate_means(load_results(v + "_ndms.pkl"), n_in) for k, (v, n_in) in eval_files.items()}

ndmsk_df = calculate_ndms_k(results_ndms)

print(ndmsk_df)

## UMWR

umwr_df = pd.DataFrame({k: unique_ratios(load_results(v + "_ndms_indices.pkl"), n_in) for k, (v, n_in) in eval_files.items()})

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

def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)

abs_distances = {}
cum_distances = {}

for k, v in test_files:
    abs_distances[k], cum_distances[k] = total_distance_dist(load_pickle(data_dir + v))

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