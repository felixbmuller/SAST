# Scene-Aware Social Transformer — 17-joint (BAM-poses) version

[[Paper]](https://arxiv.org/pdf/2409.12189) [[arXiv]](https://arxiv.org/abs/2409.12189) [[Supplementary Material]](https://owncloud.gwdg.de/index.php/s/dLwoEnW2CSnsFT9)

Implementation for the [ABAW@ECCV 24](https://affective-behavior-analysis-in-the-wild.github.io/7th/) workshop paper ["Massively Multi-Person 3D Human Motion Forecasting with Scene Context"](https://arxiv.org/pdf/2409.12189).

> **This branch (`17joints`) reproduces the paper's 17-joint results.** It uses an early version of the `hik` dataset named `bam_poses` with a 17-joint skeleton and does not depend on `hik`.
> The `main` branch is the published version, which uses `hik` with
> 29 SMPL-X joints. Use this branch if you want to evaluate or retrain the
> legacy 17-joint checkpoints

The 17 joints are, in order:

| head | arms | legs |
|------|------|------|
| 0: nose | 5: left hand | 11: left ankle |
| 1: left eye | 6: left elbow | 12: left knee |
| 2: right eye | 7: left shoulder | 13: left hip |
| 3: left ear | 8: right shoulder | 14: right hip |
| 4: right ear | 9: right elbow | 15: right knee |
| | 10: right hand | 16: right ankle |

## Usage

This code was tested with Python 3.10. Install all dependencies with

```
pip install -r requirements.txt
```

We default `OMP_NUM_THREADS` to 1.
`bam_poses` uses numba and the model uses torch, and letting both OpenMP runtimes
open a full thread pool oversubscribes the machine (on some platforms it crashes
the process outright). Parallelism comes from worker processes
(`cfg.loader.num_workers`) instead. Set the variable yourself to override:

```
OMP_NUM_THREADS=8 python train.py SAST.yaml
```

### Dataset

This branch needs the **bam_poses** dataset, please contact the author Felix Mueller to request access.

Unpack it to `data/dataset/` (or point `cfg.data.bam_location` elsewhere), so
that the directory contains one subdirectory per recording holding the pose
tracks, plus one directory per recording holding the scene geometry:

```
data/dataset/
├── A/                     # one directory per recording: A, B, C, D
│   ├── poses_pid1.npy     # (n_frames x 18 x 3) float32
│   ├── mask_pid1.npy      # (n_frames x 3) float32, [head|arms|legs] visibility
│   ├── frames_pid1.npy    # (n_frames) int64
│   ├── act_pid1.npy       # (n_frames x 82) float32, activity labels
│   └── ...                # one set of files per person id
├── A_scene/               # scene geometry, one .npy + .json per object
│   ├── collider0.npy
│   ├── collider0.json
│   └── ...
├── B/ B_scene/ C/ C_scene/ D/ D_scene/
```

The poses on disk have 18 joints; the 18th is a human-annotated location marker
near the nose and is dropped when the data is loaded.

### Preprocessing

Preprocess the dataset using

```
python sast/data/multi_person_data.py bamp SAST.yaml --splits_per_batch=256
```

This will load pose information from BAM-poses and store it at `data/bamp_[ABC]`.

**Note:** The final model in the paper uses a stride (`cfg.data.seq_offset`) of 50 frames for generating 304 frame sequences, i.e. heavily overlapping sequences. For initial experiments, I suggest using  `cfg.data.seq_offset=304` for faster extraction and training. Extracting at stride 50 requires around 100 GB of disk space.

If the extraction requires too much RAM, use a lower `--splits_per_batch`. 

You can parallelize extraction:

```
python sast/data/multi_person_data.py bamp SAST.yaml --n_shards=4 --shards=0
python sast/data/multi_person_data.py bamp SAST.yaml --n_shards=4 --shards=1
python sast/data/multi_person_data.py bamp SAST.yaml --n_shards=4 --shards=2
python sast/data/multi_person_data.py bamp SAST.yaml --n_shards=4 --shards=3
```

If you use sharding, you need to adjust `SAST.yaml` to load all shards for training:

```yaml
loader:
  dataset_parts: ["A_s0", "A_s1", "A_s2", "A_s3", "B_s0", ..., "C_s3"]
```

### Training and evaluation

Train the model with 

```
python train.py SAST.yaml
```

Generate model outputs for all sequences in the BAM-poses evaluation set (all
six activity categories of recording `D`):

```
python eval.py eval_bamp path/to/model.ckpt data/dataset/
```

This creates a file `eval.pkl` holding, per activity category, one entry per
evaluation sequence with the observed input (`seq_in`), the ground-truth future
(`seq_out_gt`) and the prediction (`seq_out_pred`). Pass `--num_processes=6` to
evaluate the six categories in parallel (one model copy per process), or
`--only_action="WHITEBOARD"` to run a single category as a smoke test.

Score the predictions with NDMS:

```
python eval.py eval_ndms eval.pkl data/dataset/
```

**Cave:** NDMS resizes every person to every other person's bone lengths and
builds one motion-word database per person. This needs a lot of scratch space in
`--tmp_dir` (~125 GB) and a lot of RAM.

`metric_calculation.py` documents how we calculate the metrics reported in the paper based on the eval files of our and baseline models. You probably need to adjust paths to files to run the script.

## Reference

If you found this repository useful, please cite

```
@inproceedings{mueller2024sast,
  author       = {Felix B. Mueller and
                  Julian Tanke and
                  Juergen Gall},
  title        = {Massively Multi-person 3D Human Motion Forecasting with Scene Context},
  booktitle    = {Computer Vision - {ECCV} 2024 Workshops - Milan, Italy, September
                  29-October 4, 2024, Proceedings, Part {XV}},
  series       = {Lecture Notes in Computer Science},
  volume       = {15637},
  pages        = {130--147},
  publisher    = {Springer},
  year         = {2024},
  url          = {https://doi.org/10.1007/978-3-031-91581-9\_10},
  doi          = {10.1007/978-3-031-91581-9\_10},
}
```
