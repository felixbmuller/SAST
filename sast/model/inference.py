import logging

import numpy as np
import torch

from bam_poses.transforms.transforms import undo_normalization_to_seq

from sast.data.constants import normalize_frame
from sast.data.multi_person_data import preprocess_sequences
from sast.data import constants


def multi_eval_fn(
    self,
    n_samples,
    persons_in,
    masks_in,
    scene,
    frame,
    n_in,
    n_out,
    pids,
    extra,
    n_out_internal=None,
    **kwargs,
):
    """
    Callback for generating results. The model predicts the data in here.
    This is the signature bam_poses.eval.Evaluation.execute() calls.

    :param persons_in: {n_persons x n_in x 17 x 3}
    :param masks_in: {n_persons x n_in}
    :param scene: {bam_poses.data.scene.Scene}
    :param frame: {int} last observed frame
    :param n_in: {int}
    :param n_out: {int} number of frames that have to be predicted
    :param pids: {List[int]}
    :param extra: {dict} additional info provided by the Evaluation, e.g. the action

    :return {n_samples x n_persons x n_out x 17 x 3}
    """

    # Avoid completely missing persons

    completely_missing = np.all(np.logical_not(masks_in), axis=-1)  # p
    if np.any(completely_missing):
        logging.error(
            f"Person {np.argmax(completely_missing)} (PID {pids[np.argmax(completely_missing)]}) at frame {frame} completely missing"
        )

        raise ValueError("completely missing persons")

    persons_in = persons_in[:, -constants.n_in :]
    masks_in = masks_in[:, -constants.n_in :]

    # Process input data

    data = preprocess_sequences(
        persons_in,
        masks_in,
        None,
        None,
        kitchen=scene.kitchen,
        object_frame=frame,
        normalize_frame=normalize_frame,
        max_persons=persons_in.shape[0],
        different_object_embeds=1,
    )

    del data["activities"]

    # Select one object embedding
    data["objects"] = data["objects"][:, 0]

    assert (
        data["primary"].shape[0] == persons_in.shape[0]
    ), f'{data["primary"].shape[0]=} == {persons_in.shape[0]=}'

    # Fake-batch and push to CUDA

    batched_data = {
        k: np.repeat(np.expand_dims(v, axis=0), n_samples, axis=0)
        for k, v in data.items()
    }

    torch_data = {
        k: torch.from_numpy(v).to(self.device) for k, v in batched_data.items()
    }

    # Predict

    # n_out needs to be devisibl by 8
    # n_out_aligned = math.ceil((n_in + n_out) / 8) * 8 - n_in
    if n_out_internal is None:
        n_out_internal = 279

    out_dict = self.forward(
        torch_data, truncate_pred_seq=True, n_out=n_out_internal, **kwargs
    )

    pred_seq = out_dict["pred_seq"].detach().cpu().numpy().astype("float32")

    pred_seq = pred_seq[:, :, :n_out]

    # Un-normalize

    n_batch, n_persons, n_frames, n_joints, n_dim = pred_seq.shape

    global_pred_seq = np.zeros(
        (n_batch, n_persons, n_frames, self.cfg.data.n_joints, n_dim)
    )  # b p t j d
    for ib in range(n_samples):
        for ip in range(pred_seq.shape[1]):

            seq = pred_seq[ib, ip]

            global_pred_seq[ib, ip] = undo_normalization_to_seq(
                seq, data["mus"][ip], data["Rs"][ip]
            )

    return global_pred_seq
