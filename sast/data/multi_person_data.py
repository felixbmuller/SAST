import sast.env  # noqa: F401  isort:skip -- must precede torch/numba imports

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

from bam_poses.data.scene import Scene
from bam_poses.data.utils import get_splits as bam_get_splits
from bam_poses.transforms.transforms import (
    normalize,
    apply_normalization_to_seq,
    apply_normalization_to_points3d,
)
from bam_poses.data.constants import activity2index

from sast.data.basis_point_representation import BasisPointSet
from sast.data.constants import (
    max_objects_per_scene,
    object_embed_dim,
    max_persons_simultaneous,
    normalize_frame,
)
from sast.utils import startup, log_exception

BASIS_POINT_SET = BasisPointSet()

# On-disk dtypes. `others` and `objects` dominate the dataset size (~95% of it),
# so they are stored as fp16 and cast back to fp32 in __getitem__. `primary` is
# the regression target and stays fp32.
STORAGE_DTYPES = {
    "primary": "float32",
    "primary_exists": "float32",
    "others": "float16",
    "others_exists": "bool",
    "objects": "float16",
    "activities": "float32",
}

# Set once per pool worker instead of being pickled with every single task.
_WORKER_KITCHEN = None


def _init_worker(kitchen):
    global _WORKER_KITCHEN
    _WORKER_KITCHEN = kitchen


def _load_scene(dataset: str, cfg) -> Scene:
    return Scene(dataset, data_location=cfg.data.bam_location)


def _memmap(save_path, name, shape):
    return np.lib.format.open_memmap(
        f"{save_path}/{name}.npy", mode="w+", shape=shape, dtype=STORAGE_DTYPES[name]
    )


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
    kitchen=None,
    object_frame=None,
    normalize_frame=None,
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
    persons: (p t 17 3)
        poses, can be either a whole sequence or only input sequence
    present: (p t)
        mask
    frames: (t)
        global frames for this sequence
    activities: (p t act)
        activities
    kitchen: Kitchen
        scene geometry. If None, the kitchen set by _init_worker() is used, which
        avoids re-pickling it for every single sequence when running in a Pool.
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

    if kitchen is None:
        kitchen = _WORKER_KITCHEN

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
    def create_to_files(
        cls,
        cfg,
        dataset,
        save_path,
        shard=0,
        n_shards=1,
        splits_per_batch=256,
    ):
        """
        Build the dataset for one scene directly into memory-mapped .npy files.

        Nothing is ever held in RAM in full: the sliding windows are cut out of
        the Scene one at a time (Scene.get_window), processed `splits_per_batch`
        at a time, and written straight to their final location. Peak memory is
        therefore set by `splits_per_batch` alone (roughly
        splits_per_batch * n_persons * 5 MB) and does not grow with seq_offset.

        Parameters
        ----------
        shard, n_shards : int
            Take only every n_shards-th window, starting at `shard`. The union over
            all shards is exactly the full set of windows, so this splits one
            dataset into n_shards independent runs (and n_shards output directories,
            to be listed together in cfg.loader.dataset_parts) without changing
            what gets generated. Use it to trade wall-clock for peak memory.
        """

        n_joints = cfg.data.n_joints
        n_embeds = cfg.data.object_embeds
        length = cfg.data.frames_in + cfg.data.frames_out

        logging.info(f"Loading scene {dataset}")

        scene = _load_scene(dataset, cfg)

        starts = bam_get_splits(
            scene.frames, length=length, stepsize=cfg.data.seq_offset
        )
        starts = starts[shard::n_shards]

        # (n_persons x n_frames), indexed like scene.frames
        exists = scene.exists_matrix()
        frame2index = scene.group.frame2index

        # Check finiteness once over the frames we are about to use, rather than
        # over a fully materialised (and heavily duplicated) array of windows.
        used_frames = np.zeros(exists.shape[1], dtype="bool")
        for start_frame in starts:
            i0 = frame2index[start_frame]
            used_frames[i0 : i0 + length] = True
        for person in scene.persons:
            cols = np.array([frame2index[f] for f in person.frames], dtype="int64")
            keep = np.all(person.masks > 0.5, axis=-1) & used_frames[cols]
            assert np.isfinite(person.poses[keep]).all(), f"pid {person.pid}"

        # The number of output rows varies per window (only persons that are
        # present at least once are kept), but it depends solely on the exists
        # matrix, which is small enough to scan up front.
        total = sum(
            int(
                np.any(
                    exists[:, frame2index[s] : frame2index[s] + length], axis=1
                ).sum()
            )
            for s in starts
        )

        logging.info(
            f"{dataset} shard {shard}/{n_shards}: {len(starts)} windows "
            f"-> {total} primary persons"
        )

        os.makedirs(save_path)

        if cfg is not None:
            with open(f"{save_path}/cfg.yaml", "w") as fp:
                fp.write(cfg.dump())

        out = {
            "primary": _memmap(save_path, "primary", (total, length, n_joints, 3)),
            "primary_exists": _memmap(save_path, "primary_exists", (total, length)),
            "others": _memmap(
                save_path,
                "others",
                (total, max_persons_simultaneous - 1, length, n_joints, 3),
            ),
            "others_exists": _memmap(
                save_path, "others_exists", (total, max_persons_simultaneous - 1)
            ),
            "objects": _memmap(
                save_path,
                "objects",
                (total, n_embeds, max_objects_per_scene, object_embed_dim),
            ),
            "activities": _memmap(
                save_path, "activities", (total, length, len(activity2index))
            ),
        }

        map_func = functools.partial(
            preprocess_sequences,
            object_frame=cfg.data.object_frame,
            normalize_frame=normalize_frame,
            different_object_embeds=n_embeds,
            return_normalization=False,
        )

        primary_min = np.full((n_joints, 3), np.inf, dtype="float32")
        primary_max = np.full((n_joints, 3), -np.inf, dtype="float32")

        written = 0

        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(
            cfg.loader.num_workers,
            initializer=_init_worker,
            initargs=(scene.kitchen,),
        ) as pool:

            for i in tqdm(range(0, len(starts), splits_per_batch)):

                tasks = []

                for s in starts[i : i + splits_per_batch]:
                    # (p t j d), (p t), (p t act) -- already in the layout
                    # preprocess_sequences expects
                    poses, present, activities = scene.get_window(s, length)
                    tasks.append(
                        (poses, present, np.arange(s, s + length), activities)
                    )

                # keep every worker busy even when the batch is small
                chunksize = max(1, min(8, len(tasks) // (cfg.loader.num_workers * 4)))

                for res in pool.starmap(map_func, tasks, chunksize=chunksize):

                    n = res["primary"].shape[0]

                    if n == 0:
                        continue

                    np.minimum(
                        primary_min, res["primary"].min(axis=(0, 1)), out=primary_min
                    )
                    np.maximum(
                        primary_max, res["primary"].max(axis=(0, 1)), out=primary_max
                    )

                    for k, arr in out.items():
                        # fp32 -> fp16 for `others` and `objects` happens here
                        arr[written : written + n] = res.pop(k)

                    written += n

                del tasks

        assert written == total, f"{written=} != {total=}"

        for arr in out.values():
            arr.flush()

        if total == 0:
            # would otherwise write +-inf and poison the range in load_from_files
            logging.warning(f"{save_path} is empty, no windows in this shard")
            primary_min[:] = 0.0
            primary_max[:] = 0.0

        # Saved so that load_from_files() does not have to stream every primary.npy
        # from disk on each start-up just to recover the normalization range.
        np.save(f"{save_path}/primary_min.npy", primary_min)
        np.save(f"{save_path}/primary_max.npy", primary_max)

        logging.info(f"Wrote {total} primary persons to {save_path}")

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
        mins = []
        maxs = []

        for dataset in parts:
            this_save_path = save_path + "_" + dataset

            dataset_instance = MultiPersonData.load_from_file(this_save_path)

            # Precomputed by create_to_files(), so we do not have to read every
            # primary.npy in full just to get the normalization range.
            mins.append(np.load(f"{this_save_path}/primary_min.npy"))
            maxs.append(np.load(f"{this_save_path}/primary_max.npy"))

            if data_mask_func is not None:

                mask = data_mask_func(
                    dataset_instance.data.activities,
                    dataset_instance.data.primary_exists,
                )

                indices = np.arange(len(dataset_instance))[mask]

                dataset_instance = data.Subset(dataset_instance, indices)

            datasets.append(dataset_instance)

        combined = data.ConcatDataset(datasets)

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
            # stored as fp16 on disk, see STORAGE_DTYPES
            "others": self.data.others[idx].astype("float32"),
            "others_exists": self.data.others_exists[idx],
            "objects": self.data.objects[idx, obj_idx].astype("float32"),
        }

        return ret

    def __len__(self):
        return self.data.primary.shape[0]


def create_dataset(
    save_name,
    cfg_path,
    datasets="ABC",
    n_shards=1,
    shards=None,
    splits_per_batch=256,
):
    """
    Parameters
    ----------
    n_shards : int
        Split each scene into this many independent runs, each covering every
        n_shards-th sliding window. The union is identical to a single run, but
        each run touches 1/n_shards of the data. Every shard becomes its own
        output directory; list them all in cfg.loader.dataset_parts.
    shards : int | tuple[int] | None
        Which shards to build in this invocation. None builds all of them
        sequentially. Pass e.g. --shards=0,1 to build a subset (useful for
        spreading the shards over several jobs).
    splits_per_batch : int
        How many sliding windows are processed before the results are flushed to
        disk. This is the only knob controlling peak memory.
    """

    cfg = startup(cfg_path)

    if shards is None:
        shards = list(range(n_shards))
    elif isinstance(shards, int):
        shards = [shards]
    else:
        shards = list(shards)

    parts = []

    for dataset in datasets:
        for shard in shards:

            part = dataset if n_shards == 1 else f"{dataset}_s{shard}"
            parts.append(part)

            save_path = f"{cfg.loader.dataset_path}/{save_name}_{part}"

            logging.info(f"Creating data {part} in {save_path}")

            MultiPersonData.create_to_files(
                cfg,
                dataset,
                save_path,
                shard=shard,
                n_shards=n_shards,
                splits_per_batch=splits_per_batch,
            )

            logging.info("Reloading data")

            data = MultiPersonData.load_from_file(save_path)

            print("length: " + str(len(data)))

            pprint({k: v.shape for k, v in data[0].items()})

            del data

    print("\nSet cfg.loader.dataset_parts to:")
    pprint(parts)


if __name__ == "__main__":
    Fire(create_dataset)
