import sast.env  # noqa: F401  isort:skip -- must precede torch/numba imports

import functools
import itertools
import logging
import multiprocessing
import os
from pathlib import Path

import torch
from fire import Fire

from bam_poses.eval import EvalNDMS, Evaluation, load_results, save_results

from sast.data.constants import EVAL_CATEGORIES
from sast.model.inference import multi_eval_fn
from sast.model.tcn import MultiTcnDiffusion
from sast.utils import startup

# the recording the models are evaluated on; A, B and C are used for training
EVAL_DATASET = "D"


def _make_evaluation(data_location, tmp_dir, n_out=250):
    return Evaluation(
        dataset=EVAL_DATASET,
        data_location=data_location,
        tmp_dir=tmp_dir,
        n_in=25,
        n_out=n_out,
    )


def _eval_category(
    ev, model_path, category, clip_range, noisy_in_seq, cache_dir, max_seq_per_action
):
    """Run the model over one evaluation category. Separate process per category."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiTcnDiffusion.load_from_checkpoint(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    logging.info(f"Loaded model, evaluating {category}")

    fn = functools.partial(
        multi_eval_fn, model, 1, clip_range=clip_range, noisy_in_seq=noisy_in_seq
    )

    return ev.execute(
        fn,
        only_action=category,
        cache_dir=cache_dir,
        max_seq_per_action=max_seq_per_action,
    )


def eval_bamp(
    model_path,
    data_location,
    out_path="eval.pkl",
    clip_range: int = 1,
    noisy_in_seq: bool = False,
    tmp_dir="tmp_sast/",
    num_processes: int = 1,
    only_action=None,
    max_seq_per_action=None,
):
    """
    Generate model predictions for every sequence of the BAM-poses evaluation set.

    Parameters
    ----------
    model_path : str
        checkpoint to evaluate
    data_location : str
        root of the BAM-poses dataset (same as cfg.data.bam_location)
    num_processes : int
        how many evaluation categories to run concurrently. Every worker loads
        its own copy of the model, so this scales memory linearly.
    only_action : str | None
        restrict the evaluation to a single category, useful for smoke tests.
        The result is *not* a complete evaluation then.
    max_seq_per_action : int | None
        only predict the first N sequences of each category. Also for smoke
        tests -- the result is not a complete evaluation.
    """

    startup(no_config=True)

    if clip_range < 0:
        clip_range = None

    model_path = Path(model_path)

    categories = EVAL_CATEGORIES if only_action is None else [only_action]

    cache_dir = f"{tmp_dir}/cache_{model_path.stem}"
    os.makedirs(cache_dir, exist_ok=True)

    ev = _make_evaluation(data_location, tmp_dir)

    logging.info("Loaded evaluation")

    if num_processes > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            all_results = pool.starmap(
                _eval_category,
                zip(
                    itertools.repeat(ev),
                    itertools.repeat(model_path),
                    iter(categories),
                    itertools.repeat(clip_range),
                    itertools.repeat(noisy_in_seq),
                    itertools.repeat(cache_dir),
                    itertools.repeat(max_seq_per_action),
                ),
            )
    else:
        all_results = [
            _eval_category(
                ev,
                model_path,
                category,
                clip_range,
                noisy_in_seq,
                cache_dir,
                max_seq_per_action,
            )
            for category in categories
        ]

    result = {}
    for r in all_results:
        result.update(r)

    save_results(out_path, result)

    logging.info(f"Wrote {out_path}")


def eval_ndms(results_path, data_location, tmp_dir="tmp_sast/", n_out=250):
    """
    Score an eval.pkl with NDMS (normalized directional motion similarity).

    Writes `<results_path stem>_ndms.pkl` and `<...>_ndms_indices.pkl` next to
    the input.

    Cave: EvalNDMS resizes every person to every other person's bone lengths and
    builds one motion-word database per person, which needs a lot of space in
    `tmp_dir` (~125 GB) and a lot of RAM.
    """

    startup(no_config=True)

    results_path = Path(results_path)

    results = load_results(results_path)

    logging.info("Creating evaluation object")

    ev = _make_evaluation(data_location, tmp_dir, n_out=n_out)

    logging.info("Creating EvalNDMS object")

    ndms = EvalNDMS(ev)

    logging.info("Running NDMS")

    avg_ndms, avg_indices = ndms.run(results)

    save_path = results_path.parent / f"{results_path.stem}_ndms.pkl"
    save_path_idxs = results_path.parent / f"{results_path.stem}_ndms_indices.pkl"

    save_results(save_path, avg_ndms)
    save_results(save_path_idxs, avg_indices)

    logging.info("done")


if __name__ == "__main__":
    Fire(
        {
            "eval_bamp": eval_bamp,
            "eval_ndms": eval_ndms,
        }
    )
