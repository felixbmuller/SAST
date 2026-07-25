from bam_poses.eval.evaluation import Evaluation
from bam_poses.data.person import Person
import bam_poses.transforms.transforms as trans
from os.path import isdir, join, isfile
from os import makedirs
from time import time
from multiprocessing.pool import ThreadPool as Pool
from typing import List
from ndms.database import Data, Database
import numpy as np
from tqdm import tqdm


class EvalNDMS:

    def __init__(self, ev: Evaluation, no_global_Rt=True) -> None:
        ev._generate_all_person_resized()  # we need all the database!
        self.ev = ev
        self.no_global_Rt = no_global_Rt

    def run(self, results, kernel_size=8, n_threads=1, n_samples=1):
        """
        :param results {
            "{action}": [
                {
                    "seq_in": {n_person x n_in x 17 x 3}
                    "seq_out_gt": {n_person x n_out x 17 x 3}
                    "seq_out_pred": {n_samples x n_person x n_out x 17 x 3}
                    "masks_in": {n_person x n_in}
                    "masks_out": {n_person x n_out}
                    "n_in": ...,
                    "nout": ...,
                    "pids": [...]
                }
            ]
        }
        """
        pid2dbname = self._build_ndms_databases(kernel_size=kernel_size)

        def create_seq(entry, sample):
            """
            :param entry: {
                "seq_in":  {n_person x n_in x 17 x 3}
                "seq_out_pred": {n_samples x n_person x n_out x 17 x 3}
            }
            """
            return np.concatenate(
                [entry["seq_in"], entry["seq_out_pred"][sample]], axis=1)

        # TODO Evaluate all 8 samples

        Avg_ndms = {}
        Avg_indices = {}
        for action, entries in (pbar := tqdm(
                results.items(), position=0, leave=True)):
            Avg_ndms[action] = []
            Avg_indices[action] = []

            data = [(create_seq(o, s), o["pids"], pid2dbname, kernel_size,
                     self.no_global_Rt)
                    for o in entries for s in range(n_samples)]

            if n_threads == 1:
                for d in tqdm(data, position=1, desc=action):
                    scores, idxs = run_ndms(*d)
                    Avg_ndms[action].append(scores)
                    Avg_indices[action].append(idxs)
            else:
                with Pool(n_threads) as p:
                    ndms_output = p.starmap(run_ndms, data)
                    assert False, "implement indices here"
                    Avg_ndms[action] = ndms_output
                    

        return Avg_ndms, Avg_indices

    def _build_ndms_databases(self, kernel_size: int):
        """
        returns:
            {
                pid: "/path/to/db.ann"
            }
        """
        db_path_for_pid = {}

        tmp_dir = join(self.ev._get_specific_tmp_dir(), "ndms")
        if not isdir(tmp_dir):
            makedirs(tmp_dir)

        load_db_from_file = []
        generate_db = []

        all_pids = [person.pid for person in self.ev.scene.persons]
        for pid in self.ev.actual_pids_in_eval:
            if self.no_global_Rt:
                db_fname = join(tmp_dir, f"db_pid{pid}_noRt.ann")
            else:
                db_fname = join(tmp_dir, f"db_pid{pid}.ann")

            assert pid not in db_path_for_pid
            db_path_for_pid[pid] = db_fname
            if not isfile(db_fname):
                generate_db.append(
                    (
                        self.ev.dataset,
                        self.ev.data_location,
                        self.ev.resized_tmp_dir,
                        db_fname,
                        kernel_size,
                        pid,
                        all_pids,
                        self.no_global_Rt
                    )
                )
        print("NDMS:")
        print("\tbuild db...")
        _start = time()
        print(f"\t* generate #{len(generate_db)} dbs...")
        if len(generate_db) > 0:
            # pool with 4 processes takes ~36GB RAM at peak
            # # and when fully utilized
            with Pool(3) as p:
                databases = p.starmap(generate_ndms_database, generate_db)
            for db_fname, (_, _, _, _, _, target_pid, _, _) in zip(databases, generate_db):  # noqa E501
                load_db_from_file.append((db_fname, kernel_size, target_pid))

        print("\t\telapsed:", time() - _start)
        return db_path_for_pid


# ===============================================
# U T I L S
# ===============================================

#   0  1  2  3  4
#   5  6  7  8  9 10
#  11 12 13 14 15 16
RELEVANT_JIDS = [0, 3, 4, 5, 6, 9, 10, 11, 12, 15, 16]


class NDMSDataset(Data):
    """"""

    @staticmethod
    def dummy_ds():
        return NDMSDataset([])

    def __init__(self, persons: List[Person]):
        self.seqs = []
        for person in persons:
            for start, end in person.get_ranges_as_frames():
                self.seqs.append(person.get_poses(
                    start, end).reshape(-1, 17 * 3))

    def __getitem__(self, index: int):
        return self.seqs[index]

    def __len__(self):
        return len(self.seqs)

    def n_dim(self):
        global RELEVANT_JIDS
        return len(RELEVANT_JIDS) * 3


def transform_data(motion_word):
    """
    :param motion_word: {kernel_size x J x 3}
    """
    global RELEVANT_JIDS
    kernel_size = len(motion_word)
    motion_word = motion_word.reshape(kernel_size, 17, 3)
    return trans.normalize(motion_word, frame=0, allow_zero_z=True)[
        :, RELEVANT_JIDS
    ].reshape(kernel_size, len(RELEVANT_JIDS) * 3)


def transform_data_noRt(motion_word):
    """
    :param motion_word: {kernel_size x J x 3}
    """
    global RELEVANT_JIDS
    kernel_size = len(motion_word)
    motion_word = motion_word.reshape(kernel_size, 17, 3)
    return trans.normalize_at_each_frame(motion_word, allow_zero_z=True)[
        :, RELEVANT_JIDS
    ].reshape(kernel_size, len(RELEVANT_JIDS) * 3)


def generate_ndms_database(
    dataset: str,
    data_location: str,
    resized_tmp_dir: str,
    db_fname: str,
    kernel_size: int,
    target_pid: int,
    all_pids: List[int],
    no_global_Rt: bool
):
    """"""
    persons = []
    for pid in all_pids:
        fname = join(resized_tmp_dir, f"pid{pid}_as_pid{target_pid}.npy")
        poses = np.load(fname)
        person = Person.load(pid=pid, dataset=dataset,
                             data_location=data_location)
        assert len(poses) == len(person), f"Nope @pid{pid}"
        person.poses = poses
        persons.append(person)
    ds = NDMSDataset(persons=persons)

    # only build the database! We cannot return
    # db though as it cannot be marshalled!
    if no_global_Rt:
        transform_data_fn = transform_data_noRt
    else:
        transform_data_fn = transform_data

    db = Database(  # noqa F841
        data=ds,
        kernel_size=kernel_size,
        transform_data_fn=transform_data_fn,
        cache_fname=db_fname,
    )
    return db_fname


def run_ndms(
        Seq,
        pids: List[int],
        pid2dbname: List[str],
        kernel_size: int,
        no_global_Rt: bool):
    """
    :param Seq: {n_person x n_times x 17 x 3}
    :param pids: [{int}]
    :param pid2dbname: [{str}]
    """
    NDMS_values = []
    Indices = []

    if no_global_Rt:
        transform_data_fn = transform_data_noRt
    else:
        transform_data_fn = transform_data()

    for i, pid in enumerate(pids):
        seq = Seq[i].astype("float64")
        fname = pid2dbname[pid]
        db = Database.load_from_cache(
            cache_fname=fname,
            kernel_size=kernel_size,
            dummy_dataset_fn=lambda: NDMSDataset.dummy_ds(),
            transform_data_fn=transform_data_fn,
        )
        framewise_ndms, indices = db.rolling_query(seq)
        NDMS_values.append(framewise_ndms)
        Indices.append(indices)

    return np.array(NDMS_values), np.array(Indices)
