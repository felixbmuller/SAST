
from collections import defaultdict
from statistics import mean
from typing import Dict, List
import numpy as np
from einops import rearrange, reduce
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


EVAL_CATEGORIES = [
    "COFFEE MACHINE",
    "WHITEBOARD",
    "FRIDGE",
    "OPEN DRAWERS AND CUPBOARDS",
    "USE SINK",
    "SITTING DOWN",
]


def pad_sequence(poses, exists):
    """
    constant pads the input sequence *in-place* with the first previous non-masked value. If the sequence is at the start, the next non-mask value
    after it is used

    poses: (t 17 3)
    exists: (t)
    """

    if np.all(exists):
        return

    if np.logical_not(np.any(exists)):
        poses[:] = np.zeros((17, 3))
        return

    filled = exists.copy()

    idx = 0

    while not np.all(filled):

        if idx > 256:
            assert False, f"{idx=} {filled=}"

        first_missing = np.argmin(filled)
        first_present = np.argmax(filled[first_missing:]) + first_missing

        # print(f"{first_missing=} {first_present=}")

        if not filled[first_present]:
            # ends with missing
            first_present = len(filled)
            # print("changed first_present! ", first_present)

        if first_missing > 0:
            filler_idx = first_missing - 1
        else:
            filler_idx = first_present

        poses[first_missing:first_present] = poses[filler_idx]

        filled[first_missing:first_present] = True

        idx += 1

IN_SEQ_LENS = {
    "MLP": 50,
    "HisRep": 50,
    "MRT": 25, 
}

def to_latex(df, highest: List[bool]):


    # apply formatting to the dataframe
    def highlight(x):
        if highest[x.name]:
            return ['bfseries: ;' if v == x.max() else ('underline:--rwrap' if v == x.nlargest(2).iloc[-1] else '') for v in x]
        else:
            return ['bfseries: ;' if v == x.min() else ('underline:--rwrap' if v == x.nsmallest(2).iloc[-1] else '') for v in x]

    df_styled = df.style.format('{:.2f}', na_rep='-').apply(highlight, axis=0)

    return df_styled.to_latex()

    # convert the styled dataframe to a LaTeX table
    #latex_table = tabulate(df_styled, headers='keys', tablefmt='latex', showindex=False)

    return latex_table


def calculate_ndms_k(means, model_keys= None, cat_key="ALL",):

    dic = defaultdict(dict)

    for name, mean in means.items():

        if model_keys is not None:
            if name not in model_keys:
                continue
            else:
                name = model_keys[name]

        v = mean[cat_key]

        dic["all"][name] = np.mean(v)

        i = 1

        while i*25 <= len(v):
            dic[f"{i}s"][name] = np.mean(v[(i-1)*25:i*25])
            i += 1


    return pd.DataFrame(dic)



def calculate_means(results, in_seq_len=0, return_all=True, ):

    if in_seq_len > 0:
        in_seq_words = in_seq_len - 8 + 1
    else:
        in_seq_words = 0

    means = {}

    all_data = []

    for k, v in results.items():

        data = np.concatenate(v, axis=-2)

        data = data[..., in_seq_words:]

        if data.ndim == 3:
            # (samples batch frames)
            data = rearrange(data, "s b t -> (s b) t")

        #means[k] = reduce(data, "b t -> t", "mean")
        means[k] = np.nanmean(data, axis=0)
        all_data.append(data)

    if return_all:
        data = np.concatenate(all_data, axis=0)

        means["ALL"] = np.nanmean(data, axis=0)
        #means["ALL"] = reduce(data, "b t -> t", "mean")

    return means

def unique_ratios(results, in_seq_len=25):

    in_seq_words = in_seq_len - 8 + 1

    unique_rate = defaultdict(list)

    for k, v in results.items():
        for arr in v:
            for flat_arr in arr:

                unique_rate["all"].append(len(np.unique(flat_arr[in_seq_words:])) / len(flat_arr[in_seq_words:]))

                i = 1

                while in_seq_words+i*25 <= len(flat_arr):
                    section = flat_arr[in_seq_words+(i-1)*25:in_seq_words+i*25]
                    ratio = len(np.unique(section))/len(section)

                    unique_rate[f"{i}s"].append(ratio)

                    i += 1

    ratios = {k : mean(v) for k, v in unique_rate.items()}

    return ratios

def get_velocity(results, gt=False):


    vels = defaultdict(list)

    printed_n_in = False

    for k, v in results.items():
        for sample in v:

            if not printed_n_in:
                print("n_in ", sample["n_in"])
                printed_n_in = True

            if gt:
                seq = sample["seq_out_gt"]
            else:
                seq = sample["seq_out_pred"][0]

            seq = seq.copy()

            for i in range(seq.shape[0]):
                pad_sequence(seq[i], sample["masks_out"][i, :seq.shape[1]])

            #seq = np.concatenate([sample["seq_in"][:, -1:], seq], axis=1) # p t j d

            global_directional_movement = np.mean(seq[..., [13, 14], :][..., :2], axis=-2)

            diff = np.diff(global_directional_movement, axis=-2) # directional velocity

            vel = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2) * 25# net velocity

            vel = np.clip(vel, None, 10)

            #vel[np.logical_not(sample["masks_out"][:, 1:vel.shape[1]+1])] = np.nan

            vels[k].append(vel)

    return vels

def get_local_velocity(results, gt=False):


    vels = defaultdict(list)

    printed_n_in = False

    for k, v in results.items():
        for sample in v:

            if not printed_n_in:
                print("n_in ", sample["n_in"])
                printed_n_in = True

            seq = sample["seq_out_pred" if not gt else "seq_out_gt"]

            global_mean_pose = np.mean(seq[..., [13, 14], :], axis=-2)

            seq = seq - global_mean_pose[..., np.newaxis, :]

            diff = np.diff(seq, axis=-3) # directional velocity

            vel = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2 + diff[..., 2]**2) # net velocity

            vel = np.mean(vel, axis=-1) # mean over joints

            vels[k].append(vel)

    return vels


def plot_means_per_frame(results : Dict[str, dict], keys=None, plot_all=False, ylabel="NDMS", ylim=None, save=None, figsize=(16,9)):

    baselines = ["TriPod", "SiMLPe", "HisRep", "MRT"]

    colors = {
        "Ours (Undersampled)": "orange",
        "Ours": "green",
        "Ours (Normal)": "red",
        "Ours (AllData)": "orange"
    }

    sns.set_style()

    if plot_all:
        fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, figsize=figsize)
        categories = ["ALL"]

        axs = [axs]
    else:

        fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(16, 9))

        categories = EVAL_CATEGORIES

        axs = [*axs[0], *axs[1]]

    for i, cat in enumerate(categories):

        for name, d in results.items():

            if keys is not None:
                if name not in keys:
                    continue
                else:
                    name = keys[name]

            if name in baselines:
                style = "--"
            else:
                style= "-"

            color = None
            if name in colors:
                color = colors[name]

            axs[i].plot(d[cat], style, color=color, label=name)

        #axs[i].axhline(y=0.27, linestyle='--', label="Real Other Dataset (.27)")
        if not plot_all:
            axs[i].set_title(cat)

        axs[i].set_ylabel(ylabel)
        axs[i].set_xlabel("output frames")

        
        if ylim is not None:
            axs[i].set_ylim(0.0, ylim)
        #axs[i].set_xlim(0, 250)

    fig.tight_layout()
    plt.legend()
    if save is not None:
        plt.savefig(save)
    plt.show()