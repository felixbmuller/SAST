import functools
import pickle
from pathlib import Path
from fire import Fire

from hik.eval import Evaluator

from sast.model.inference import multi_eval_fn
from sast.model.tcn import MultiTcnDiffusion
from sast.utils import (
    startup,
)


def eval_hik(
    model_path,
    data_path,
    clip_range: int = 1,
    noisy_in_seq: bool = False,
    device="cuda",
):
    data_path = Path(data_path)

    startup(no_config=True)

    if clip_range < 0:
        clip_range = None

    model_path = Path(model_path)

    EVAL_DATASET = "D"  # choose the evaluation dataset
    ev = Evaluator(
        data_path / "hik_test.json", dataset=EVAL_DATASET, data_path=data_path
    )

    model = MultiTcnDiffusion.load_from_checkpoint(model_path).to(device)
    model.eval()

    fn = functools.partial(
        multi_eval_fn, model, 1, clip_range=clip_range, noisy_in_seq=noisy_in_seq
    )

    result = ev.execute3d(fn)

    with open("eval.pkl", "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    Fire(eval_hik)
