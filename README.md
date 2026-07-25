# Scene-Aware Social Transformer

[[Paper]](https://arxiv.org/pdf/2409.12189) [[arXiv]](https://arxiv.org/abs/2409.12189) [[Supplementary Material]](https://owncloud.gwdg.de/index.php/s/dLwoEnW2CSnsFT9)

Implementation for the [ABAW@ECCV 24](https://affective-behavior-analysis-in-the-wild.github.io/7th/) workshop paper ["Massively Multi-Person 3D Human Motion Forecasting with Scene Context"](https://arxiv.org/pdf/2409.12189).

## Usage

This code was tested with Python 3.10. Install all dependencies with

```
pip install -r requirements.txt
```

Download the [Humans in Kitchens](https://github.com/jutanke/hik/tree/main) and unpack its content to `data/`, such that `data/` contains `poses/`, `scenes/`, and `body_models/`.

### Preprocessing

Preprocess the dataset using

```
python sast/data/multi_person_data.py hik SAST.yaml --splits_per_batch=256
```

This will load pose information from Humans in Kitchens and store them at `data/hik_[ABC]`.

**Note:** The final model in the paper uses a stride (`cfg.data.seq_offset`) of 50 frames for generating 304 frame sequences, i.e. heavily overlapping sequences. For initial experiments, I suggest using  `cfg.data.seq_offset=304` for faster extraction and training. Extracting at stride 50 requires around 100 GB of disk space.

If the extraction requires too much RAM, use a lower `--splits_per_batch`. 

You can parallelize extraction:

```
python sast/data/multi_person_data.py hik SAST.yaml --n_shards=4 --shards=0
python sast/data/multi_person_data.py hik SAST.yaml --n_shards=4 --shards=1
python sast/data/multi_person_data.py hik SAST.yaml --n_shards=4 --shards=2
python sast/data/multi_person_data.py hik SAST.yaml --n_shards=4 --shards=3
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

Generate model outputs for all sequences in the Humans in Kitchens evaluation set using `hik.eval.Evaluator`.

```
python eval.py path/to/model data/
```

This will create a file `eval.pkl` that can be analyzed using Humans in Kitchens evaluation code. 

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
