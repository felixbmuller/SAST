import itertools
from types import SimpleNamespace
import logging
import multiprocessing
from pprint import pprint
import os
import functools

import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from fire import Fire
from einops import rearrange

from hik.data.scene import Scene
from hik.transforms.transforms import (
    normalize,
    apply_normalization_to_seq,
    apply_normalization_to_points3d,
)
from hik.data.constants import activity2index

from sast.data.basis_point_representation import BasisPointSet
from sast.data.constants import (
    max_objects_per_scene,
    object_embed_dim,
    max_persons_simultaneous,
    normalize_frame,
)
from sast.utils import startup, log_exception

BASIS_POINT_SET = BasisPointSet()


def _load_dataset(dataset: str, cfg):
    data_location = cfg.data.hik_location

    scene = Scene.load_from_paths(
        dataset,
        data_location + "/poses/",
        data_location + "/scenes/",
        data_location + "/body_models/",
    )

    n_in = cfg.data.frames_in
    n_out = cfg.data.frames_out

    splits = scene.get_splits(
        n_in + n_out,
        stepsize=cfg.data.seq_offset,
    )

    poses = rearrange(splits["poses3d"], "b t p j d -> b p t j d")
    masks = rearrange(splits["masks"], "b t p -> b p t")
    start_frames = splits["start_frames"]
    activities = rearrange(splits["activities"], "b t p act -> b p t act")

    frames = [
        np.arange(start_frame, start_frame + n_in + n_out)
        for start_frame in start_frames
    ]

    assert np.isfinite(poses[masks > 0.99]).all()

    logging.info(f"Loaded dataset {dataset} with shape {poses.shape}")

    return poses, masks, frames, activities, scene


def pad_sequence(poses, exists):
    """
    constant pads the input sequence *in-place* with the first previous non-masked value. If the sequence is at the start, the next non-mask value
    after it is used

    poses: (t 29 3)
    exists: (t)
    """

    if np.all(exists):
        return

    if np.logical_not(np.any(exists)):
        poses[:] = np.zeros((29, 3))
        return

    filled = exists.copy()

    idx = 0

    while not np.all(filled):

        if idx > 256:
            assert False, f"{idx=} {filled=}"

        first_missing = np.argmin(filled)
        first_present = np.argmax(filled[first_missing:]) + first_missing

        if not filled[first_present]:
            # ends with missing
            first_present = len(filled)

        if first_missing > 0:
            filler_idx = first_missing - 1
        else:
            filler_idx = first_present

        poses[first_missing:first_present] = poses[filler_idx]

        filled[first_missing:first_present] = True

        idx += 1


def preprocess_sequences(
    persons,
    present,
    frames,
    activities,
    kitchen,
    object_frame,
    normalize_frame,
    max_persons=max_persons_simultaneous,
    different_object_embeds=8,
    return_normalization=True,
    progress_indicator=False,
):
    """

    Cave
    ----
    The returned number of persons is the number of unpadded persons present in the scene.
    This can vary for each call of preprocess_sequences. Do not try to stack returns of this
    functions, concatenate them (the person axis can be merged with the batch axis, as there is
    not information flow inbetween in the model).

    The act

    Masking Policy
    ---------------
    - Persons which are completely missing are zero-padded and are ignored in the transformer and where else possible
    - Persons which are partially present are constant-padded everywhere where they are missing. They are trated as fully present persons everywhere except in the loss function. The model is only judged on the loss on the parts of the sequence that were always present

    Parmeters
    ---------
    persons: (p t 29 3)
        poses, can be either a whole sequence or only input sequence
    present: (p t)
        mask
    frames: (t)
        global frames for this sequence
    activities: (p t act)
        activities
    object_frame: int
        local frame where to extract objects. If frames=None, this is assumed to be the global frame instead.
    normalize_frame: int
        frame in pose sequence (local) to be used for normalization

    Returns
    -------
    primary : Tensor(p t j d)
        normalized sequences for primary persons
    primary_exists : FloatTensor(p t)
        True if the other person if partially or fully present, i.e. not padding
    others : Tensor(p o t j d)
        normalized (w.r.t. primary person) sequences of other persons, excluding primary person!
    others_exists : BoolTensor(p o)
        True if the other person if partially or fully present, i.e. not padding
    mus : Tensor(person 3)
        translation used for normalization
    Rs: Tensor(person 3 3)
        rotation used for normalization
    objects : Tensor(p different_object_embeds obj embed)
        embedded objects. This contains `different_object_embeds` different random embeddings to avoid overfitting
    activities : Tensor(p t act)
        activities
        CAVE: The activities for all present frames are sequeezed together. The activities do not correspond to the correct frames anyome.
    """

    if progress_indicator:
        print(".", end="", flush=True)

    n_persons, n_frames, n_joints, n_dim = persons.shape

    persons = persons.copy()

    # pad sequences
    for ip in range(n_persons):
        pad_sequence(persons[ip], present[ip])

    present_agg = np.any(present, axis=-1)  # (p)

    unpadded_n_persons = np.sum(present_agg)

    n_primary_joints = n_joints

    primary = np.zeros(
        (unpadded_n_persons, n_frames, n_primary_joints, n_dim), dtype="float32"
    )
    others = np.zeros(
        (unpadded_n_persons, max_persons - 1, n_frames, n_joints, n_dim),
        dtype="float32",
    )

    primary_exists = np.zeros((unpadded_n_persons, n_frames), dtype="float32")
    others_exists = np.zeros((unpadded_n_persons, max_persons - 1), dtype="bool")

    mus = np.zeros((unpadded_n_persons, n_dim), dtype="float32")
    Rs = np.zeros((unpadded_n_persons, n_dim, n_dim), dtype="float32")

    objects = np.zeros(
        (
            unpadded_n_persons,
            different_object_embeds,
            max_objects_per_scene,
            object_embed_dim,
        ),
        dtype="float32",
    )

    if activities is not None:
        activities = activities[present_agg]

    out_idx = 0

    global_frame = object_frame if frames is None else frames[object_frame]

    objs_raw = kitchen.get_environment(global_frame)

    for ip in range(n_persons):
        if not present_agg[ip]:
            continue

        norm_seq, (mu, R) = normalize(
            persons[ip], normalize_frame, return_transform=True
        )

        primary[out_idx] = norm_seq
        mus[out_idx] = mu
        Rs[out_idx] = R

        primary_exists[out_idx] = present[ip]

        # Process others

        inner_idx = 0

        for jp in range(n_persons):
            if not present_agg[jp] or jp == ip:
                continue

            others[out_idx, inner_idx] = apply_normalization_to_seq(persons[jp], mu, R)
            others_exists[out_idx, inner_idx] = True

            inner_idx += 1

        # Process objects

        for obj_batch in range(different_object_embeds):

            objs_norm_pointclouds = [
                apply_normalization_to_points3d(obj.query(), mu, R) for obj in objs_raw
            ]

            objs_labels = [obj.label for obj in objs_raw]

            this_objs = create_objects(objs_norm_pointclouds, objs_labels)

            objects[out_idx, obj_batch] = this_objs

        out_idx += 1

    ret = dict(
        primary=primary,
        primary_exists=primary_exists,
        others=others,
        others_exists=others_exists,
        objects=objects,
        activities=activities,
    )

    if return_normalization:
        ret["mus"] = mus
        ret["Rs"] = Rs

    return ret


def create_objects(objs_norm_pointclouds, objs_labels):
    """
    Returns: (o e)
    """

    obj_feats = [
        np.concatenate([BASIS_POINT_SET.query(norm_obj), label])
        for norm_obj, label in zip(objs_norm_pointclouds, objs_labels)
    ]

    embed_size = len(obj_feats[0])

    padding = [np.zeros((embed_size,), dtype=np.float32)] * (
        max_objects_per_scene - len(obj_feats)
    )

    objects = np.stack(obj_feats + padding)

    return objects


class MultiPersonData(data.Dataset):

    @classmethod
    def load_from_hik(
        cls,
        cfg,
        load_only=None,
    ):
        """ """

        logging.info("Loading datasets")

        with multiprocessing.get_context("spawn").Pool(cfg.loader.num_workers) as pool:
            # if True:

            print("workers", cfg.loader.num_workers)

            if load_only is not None:
                datasets = [_load_dataset(load_only, cfg)]

            else:

                datasets = pool.starmap(
                    _load_dataset,
                    (("A", cfg), ("B", cfg), ("C", cfg)),
                )

            logging.info("Preprocess sequences in parallel")

            total = sum(d[0].shape[0] for d in datasets)

            data_iter = itertools.chain.from_iterable(
                (
                    zip(
                        poses,
                        masks,
                        frames,
                        activities,
                        itertools.repeat(scene.kitchen),
                    )
                    for poses, masks, frames, activities, scene in datasets
                )
            )

            map_func = functools.partial(
                preprocess_sequences,
                object_frame=cfg.data.object_frame,
                normalize_frame=normalize_frame,
                return_normalization=False,
                progress_indicator=True,
            )

            sequences = pool.starmap(map_func, data_iter, chunksize=100)

        del datasets

        self = SimpleNamespace()

        logging.info("Concatenating primary")
        self.primary = np.concatenate([r.pop("primary") for r in sequences])
        logging.info("Concatenating primary_exists")
        self.primary_exists = np.concatenate(
            [r.pop("primary_exists") for r in sequences]
        )
        logging.info("Concatenating others")
        self.others = np.concatenate([r.pop("others") for r in sequences])
        logging.info("Concatenating others_exists")
        self.others_exists = np.concatenate([r.pop("others_exists") for r in sequences])

        logging.info("Concatenating objects")
        self.objects = np.concatenate([r.pop("objects") for r in sequences])

        logging.info("Concatenating activities")
        self.activities = np.concatenate([r.pop("activities") for r in sequences])

        logging.info(
            "dataset size (number of primary persons) " + str(self.primary.shape[0])
        )

        del sequences

        logging.info("Calculate metrics")
        if load_only is None:
            self.data_mean = np.mean(self.primary, axis=(0, 1))
            self.data_std = np.std(self.primary, axis=(0, 1))

        logging.info("Data loading complete")

        return cls(self, cfg)

    @classmethod
    def load_from_file(cls, save_path):

        keys = [
            "primary",
            "primary_exists",
            "others",
            "others_exists",
            "objects",
            "activities",
        ]

        data = {}

        for k in tqdm(keys):
            data[k] = np.load(f"{save_path}/{k}.npy", mmap_mode="c")

        return cls(SimpleNamespace(**data), None)

    @classmethod
    def load_from_files(cls, save_path, parts, data_mask_func=None):

        datasets = []

        for dataset in parts:
            this_save_path = save_path + "_" + dataset

            dataset_instance = MultiPersonData.load_from_file(this_save_path)

            if data_mask_func is not None:

                mask = data_mask_func(
                    dataset_instance.data.activities,
                    dataset_instance.data.primary_exists,
                )

                indices = np.arange(len(dataset_instance))[mask]

                dataset_instance = data.Subset(dataset_instance, indices)

            datasets.append(dataset_instance)

        combined = data.ConcatDataset(datasets)

        mins = []
        maxs = []

        for dataset in datasets:
            mins.append(np.min(dataset.dataset.data.primary, axis=(0, 1)))
            maxs.append(np.max(dataset.dataset.data.primary, axis=(0, 1)))

        global_min = np.min(np.stack(mins), axis=0)
        global_max = np.max(np.stack(maxs), axis=0)

        # Fake mean and std for normal scaling
        mean = (global_min + global_max) / 2
        std = (global_max - global_min) / 6

        combined.get_mean = lambda: mean
        combined.get_std = lambda: std

        return combined

    def __init__(self, data: SimpleNamespace, cfg):

        self.data = data
        self.cfg = cfg

    def get_mean(self):
        return self.data.data_mean

    def get_std(self):
        return self.data.data_std

    def save(self, save_path):

        os.makedirs(save_path)

        if self.cfg is not None:
            with open(f"{save_path}/cfg.yaml", "w") as fp:
                fp.write(self.cfg.dump())

        for k, v in tqdm(vars(self.data).items()):
            np.save(f"{save_path}/{k}.npy", v)

    def __getitem__(self, idx):
        """

        Returns
        -------
        objects: np.array(o e)

        see preprocess_sequences() for how the other returns look like (just without the 'p'
        axis)
        """

        obj_idx = np.random.randint(0, self.data.objects.shape[1])

        ret = {
            "primary": self.data.primary[idx],
            "primary_exists": self.data.primary_exists[idx],
            "others": self.data.others[idx],
            "others_exists": self.data.others_exists[idx],
            "objects": self.data.objects[idx, obj_idx],
        }

        return ret

    def __len__(self):
        return self.data.primary.shape[0]


def create_dataset(save_name, cfg_path, datasets="ABC"):

    cfg = startup(cfg_path)

    for dataset in datasets:
        # if True:

        logging.info("Loading data " + dataset)

        data = MultiPersonData.load_from_hik(cfg, load_only=dataset)

        logging.info("Saving data")

        save_path = f"{cfg.loader.dataset_path}/{save_name}_{dataset}"

        data.save(save_path)

        logging.info("Reloading data")

        del data

        data = MultiPersonData.load_from_file(save_path)

        logging.info("Reloaded data")

        print("length: " + str(len(data)))

        pprint({k: v.shape for k, v in data[0].items()})

        del data


if __name__ == "__main__":
    Fire(create_dataset)
