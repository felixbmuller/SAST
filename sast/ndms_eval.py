"""NDMS evaluation on 29-joint Humans-in-Kitchens (hik) data.

The heavy lifting is done by :class:`hik.eval.longterm.NDMSData`, which builds a
per-target-person motion-word database by re-rendering every other person's SMPL
motion onto the target person's body shape.

Output:

    run(results) -> (avg_ndms, avg_indices)

where both are ``{action: [np.ndarray(n_person, n_words)]}`` dicts, one entry per
sample in ``results[action]`` and ``n_words = n_frames - kernel_size + 1``.
"""

import logging
from os import makedirs
from os.path import isdir, join

import numpy as np
from tqdm import tqdm

from ndms.database import Database
from hik.data import PersonSequences
from hik.data.smpl import Body
from hik.eval.longterm import NDMSData, transform as ndms_transform

# Joints kept for NDMS, in hik's 29-joint layout. This MUST stay in sync with
# ``hik.eval.longterm.NDMSData.selected_jids`` -- the database stores poses
# reduced to exactly these joints, so queries have to be reduced the same way.
NDMS_SELECTED_JIDS = [0, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21, 26, 27]

# Fixed at 8: sast.metrics trims the input-sequence motion words assuming a
# kernel size of 8 (``in_seq_words = in_seq_len - 8 + 1``).
KERNEL_SIZE = 8


class NDMSEvaluator:
    """Compute NDMS scores + matched-word indices for forecasting results.

    :param dataset: one of ``{"A", "B", "C", "D"}``
    :param data_path: root of the hik dataset, containing ``poses/`` and
        ``body_models/`` (same layout ``eval.py`` passes to ``hik.eval.Evaluator``)
    :param cache_dir: where rendered sequences and the Annoy databases are cached
    :param kernel_size: motion-word length; keep at 8 to match ``sast.metrics``
    """

    def __init__(
        self, dataset, data_path, cache_dir="tmp_sast/", kernel_size=KERNEL_SIZE
    ):
        self.dataset = dataset
        self.kernel_size = kernel_size
        self.cache_dir = join(cache_dir, f"ndms/{dataset}")
        if not isdir(self.cache_dir):
            makedirs(self.cache_dir)

        logging.info("Loading hik person sequences for dataset %s", dataset)
        self.person_seqs = PersonSequences(
            person_path=join(data_path, "poses")
        ).get_sequences(dataset=dataset)
        self.body = Body(smplx_path=join(data_path, "body_models"))

        # Databases are keyed by target pid and are independent of the results
        # being scored, so we build each one lazily and reuse it across files.
        self._databases = {}

    def _get_database(self, pid):
        if pid not in self._databases:
            logging.info("Building NDMS database for pid %s", pid)
            data = NDMSData(
                self.person_seqs, pid, self.body, cache_dir=self.cache_dir
            )
            self._databases[pid] = Database(
                data=data,
                kernel_size=self.kernel_size,
                transform_data_fn=ndms_transform,
                cache_fname=join(self.cache_dir, f"db_pid{pid}.ann"),
            )
        return self._databases[pid]

    def run(self, results):
        """Score forecasting results.

        :param results: ``{action: [entry]}`` where each entry holds
            ``seq_in`` ``(n_person, n_in, 29, 3)``,
            ``seq_out_pred`` ``(n_samples, n_person, n_out, 29, 3)`` and
            ``pids`` ``[int]``.
        :returns: ``(avg_ndms, avg_indices)``, see module docstring.
        """
        avg_ndms = {}
        avg_indices = {}

        for action, entries in tqdm(results.items(), position=0, leave=True):
            ndms_per_entry = []
            indices_per_entry = []

            for entry in entries:
                # Observed input followed by the (first) predicted sample.
                seq = np.concatenate(
                    [entry["seq_in"], entry["seq_out_pred"][0]], axis=1
                )  # (n_person, n_total, 29, 3)
                pids = entry["pids"]

                ndms_values = []
                index_values = []
                for i, pid in enumerate(pids):
                    # Reduce to the NDMS joints so the query matches the
                    # database's feature space, then roll the kernel over it.
                    query = seq[i][:, NDMS_SELECTED_JIDS].astype("float64")
                    db = self._get_database(pid)
                    framewise_ndms, indices = db.rolling_query(query)
                    ndms_values.append(framewise_ndms)
                    index_values.append(indices)

                ndms_per_entry.append(np.array(ndms_values))
                indices_per_entry.append(np.array(index_values))

            avg_ndms[action] = ndms_per_entry
            avg_indices[action] = indices_per_entry

        return avg_ndms, avg_indices
