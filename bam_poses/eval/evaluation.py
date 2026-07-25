from tqdm import tqdm
from time import time
from multiprocessing import Pool
import numpy as np
from os import makedirs
from os.path import isdir, isfile, join
import bam_poses.data.constants as const
from bam_poses.data.utils import frames2segments, get_tmp_dir
from bam_poses.data.scene import Scene
import pickle
from bam_poses.eval.utils import save_results


ACTIVITIES_FOR_EVAL = {
    "COFFEE MACHINE": [
        "place cup onto coffee machine",
        "take cup from coffee machine",
        "fill coffee beens",
        "make coffee",
        "press coffee button",
        "fill water to coffee machine",
        "place water tank in coffee machine",
        "empty ground from coffee machine",
        "take water tank from coffee machine",
        "empty water from coffee machine",
    ],
    "WHITEBOARD": [
        "carry whiteboard eraser",
        "carry whiteboard marker",
        "draw on whiteboard",
        "erase on whiteboard",
        "place sheet onto whiteboard",
        "remove sheet from whiteboard",
    ],
    # "CAKE": [
    #     "place cake on table",
    #     "cut cake in pieces",
    #     "place cake on plate"
    # ],
    "FRIDGE": [
        "open fridge",
        "put cake in fridge",
        "take cake out of fridge",
        "close fridge",
    ],
    "OPEN DRAWERS AND CUPBOARDS": [
        "open drawer",
        "open cupboard",
        "close cupboard",
        "close drawer",
    ],
    "USE SINK": [
        "clean dish",
        "empty cup in sink",
        "put water in glass",
        "put water in kettle",
        "washing hands",
    ],
    "SITTING DOWN": ["sitting down"],
}


def _get_start_frames(scene, activities, tmp_file=None):
    """"""
    if tmp_file is not None and isfile(tmp_file):
        with open(tmp_file, "rb") as f:
            return pickle.load(f)
    else:
        frames = []
        for act in activities:
            F = scene.group.get_frames_where(min_persons=1, activities=[act])
            S = frames2segments(F, include_length=False)
            for frame, _ in S:
                frames.append(frame)
        if tmp_file is not None:
            with open(tmp_file, "wb") as f:
                pickle.dump(frames, f)

        return frames


class Evaluation:
    def __init__(
        self,
        dataset: str,
        data_location: str,
        n_in=25,
        n_out=250,
        tmp_dir=get_tmp_dir(),
    ) -> None:
        self.tmp_dir = tmp_dir
        self.n_in = n_in
        self.n_out = n_out
        self.dataset = dataset
        self.data_location = data_location
        self.scene = Scene(
            dataset=dataset, data_location=data_location, default_to_box=False
        )
        self._generate_all_evalframes()
        # self._generate_all_person_resized()

    def _get_specific_tmp_dir(self):
        return join(self.tmp_dir, f"bam_poses_eval/{self.dataset}")

    def _generate_all_evalframes(self):
        """
        generates all the evaluation frames for this eval
        """
        self.eval2frames = {}
        self.eval_pids = []
        tmp_dir = self._get_specific_tmp_dir()
        if not isdir(tmp_dir):
            makedirs(tmp_dir)
        All_Frames = []
        for eval_name, activities in ACTIVITIES_FOR_EVAL.items():
            tmp_file = join(tmp_dir, eval_name.replace(" ", "_") + ".pkl")
            F = _get_start_frames(
                self.scene, activities=activities, tmp_file=tmp_file)
            # --
            F = [f for f in F if f < const.LAST_FRAMES[self.dataset]]
            All_Frames += F
            self.eval2frames[eval_name] = F

        actual_pids_in_eval = set()
        All_Frames = list(set(All_Frames))
        for frame in All_Frames:
            for person in self.scene.persons:
                if person.has_frame(frame):
                    actual_pids_in_eval.add(person.pid)
        self.actual_pids_in_eval = list(actual_pids_in_eval)

    def _generate_all_person_resized(self):
        """
        resizes all persons into each others sizes. This will make it much
        faster for eevaluation
        """
        print("\tgenerate all person sizes...")
        _start = time()
        self.person2personsize = {}  # (pid_real, pid_size) -> Person
        tmp_dir = self._get_specific_tmp_dir()
        tmp_dir = join(tmp_dir, "resized")
        if not isdir(tmp_dir):
            makedirs(tmp_dir)

        self.resized_tmp_dir = tmp_dir

        # for person1 in self.scene.persons:
        Conversions_that_have_to_be_run = []  # (fname, person, person_target)
        for person1 in self.scene.persons:
            for person2 in self.scene.persons:
                fname = join(
                    tmp_dir, f"pid{person1.pid}_as_pid{person2.pid}.npy")
                if person1.pid == person2.pid:
                    if not isfile(fname):
                        np.save(fname, person1.poses)
                else:
                    if not isfile(fname):
                        Conversions_that_have_to_be_run.append(
                            (fname, person1, person2)
                        )
                # ---)

        if len(Conversions_that_have_to_be_run) > 0:
            Adjusted_Poses3d = []
            print(f"\t\thandle #{len(Conversions_that_have_to_be_run)}")
            with Pool(8) as p:
                Adjusted_Poses3d += p.starmap(
                    adjust_length, Conversions_that_have_to_be_run
                )
        print("\t\telapsed:", time() - _start)


    def execute(self, eval_fn, pass_future_gt=False, max_seq_per_action=None, only_action=None, only_seq=None, cache_dir=None):
        """
        Runs and saves the model results
        :param eval_fn: {function}
            def eval_fn(persons_in, masks_in, scene, frame, n_in, n_out, pids, extra)  # noqa E501
                -> persons_out
                    Important: {persons_out} has to be "sampled:
                        n_samples x n_persons x n_out x 17 * 3
        """
        Result = {}
        for idx, (action, frames) in enumerate(self.eval2frames.items()):
            if only_action is not None and action != only_action:
                continue 

            Result[action] = []
            for j, frame in enumerate(tqdm(frames[:max_seq_per_action], position=idx, desc=str(action))):
                if only_seq is not None and j != only_seq:
                    Result[action].append(dict())
                    continue

                if j > 0 and j % 10 == 0 and cache_dir is not None:
                    save_results(f"{cache_dir}/{action}_{j:04}.pkl", Result[action])

                Poses_in = []
                Masks_in = []
                Poses_out = []
                Masks_out = []
                Pids = []
                for person in self.scene.get_persons_at_frame(frame):
                    poses, masks = person.try_get_poses(
                        start_frame=frame - self.n_in, end_frame=frame + self.n_out
                    )
                    poses_in = poses[: self.n_in]
                    masks_in = masks[: self.n_in]
                    if np.sum(masks_in) > 0:
                        poses_out = poses[self.n_in:]
                        masks_out = masks[self.n_in:]
                        Poses_in.append(poses_in)
                        Masks_in.append(masks_in)
                        Poses_out.append(poses_out)
                        Masks_out.append(masks_out)
                        Pids.append(person.pid)

                Poses_out = np.array(Poses_out, dtype=np.float32)
                Poses_in = np.array(Poses_in, dtype=np.float32)
                Masks_in = np.array(Masks_in, dtype=np.float32)
                Masks_out = np.array(Masks_out, dtype=np.float32)

                n_person = len(Poses_in)

                extra = {"action": action}
                if pass_future_gt:
                    extra["poses_out"] = Poses_out

                Poses_out_hat = eval_fn(
                    Poses_in,
                    Masks_in,
                    self.scene,
                    frame,
                    self.n_in,
                    self.n_out,
                    Pids,
                    extra,
                )

                # Poses_out_hat <= n_samples x n_persons x n_out x 17 * 3

                if not isinstance(Poses_out_hat, np.ndarray):
                    raise ValueError(
                        "The returned motion must be a numpy array!")

                if Poses_out_hat.shape[1] != n_person:
                    raise ValueError(
                        "Must provide the same amount of persons as was input"
                    )
                if Poses_out_hat.shape[2] != self.n_out:
                    raise ValueError(
                        f"Must generate exactly {self.n_out} frames")

                if len(Poses_out_hat.shape) == 4:
                    Poses_out_hat = Poses_out_hat.reshape(
                        n_person, self.n_out, 17, 3)
                elif len(Poses_out_hat.shape) == 5:
                    if (
                        Poses_out_hat.shape[3] != 17 or Poses_out_hat.shape[4] != 3
                    ):  # noqa E501
                        raise ValueError(f"Weird shape: {Poses_out_hat.shape}")
                else:
                    raise ValueError(
                        f"Unknown Poses_out_hat shape: {Poses_out_hat.shape}"
                    )

                Result[action].append(
                    {
                        "seq_in": Poses_in,
                        "seq_out_gt": Poses_out,
                        "seq_out_pred": Poses_out_hat,
                        "masks_in": Masks_in,
                        "masks_out": Masks_out,
                        "frame": frame,
                        "action": action,
                        "dataset": self.dataset,
                        "n_in": self.n_in,
                        "n_out": self.n_out,
                        "pids": Pids,
                    }
                )
        return Result


def adjust_length(fname, person1, person2):
    """
    adjust person1 bone length to person2's bones
    """
    person1_as_2 = person1.adjust_pose_lengths_like(person2)
    np.save(fname, person1_as_2.poses)
    return fname
