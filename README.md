# Scene-Aware Social Transformer

[[Paper]](https://arxiv.org/pdf/2409.12189) [[arXiv]](https://arxiv.org/abs/2409.12189) [[Supplementary Material]](https://owncloud.gwdg.de/index.php/s/dLwoEnW2CSnsFT9)

Implementation for the [ABAW@ECCV 24](https://affective-behavior-analysis-in-the-wild.github.io/7th/) workshop paper ["Massively Multi-Person 3D Human Motion Forecasting with Scene Context"](https://arxiv.org/pdf/2409.12189).

## Usage

This code was tested with Python 3.10. Install all dependencies with

```
pip install -r requirements.txt
```

Download the [Humans in Kitchens](https://github.com/jutanke/hik/tree/main) and unpack its content to `data/`, such that `data/` contains `poses/`, `scenes/`, and `body_models/`.

Preprocess the dataset using

```
python sast/data/multi_person_data.py hik SAST.yaml 
```

This will load pose information from Humans in Kitchens and store them at `data/hik_[ABC]`.

Train the model with 

```
python train.py SAST.yaml
```

Generate model outputs for all sequences in the Humans in Kitchens evaluation set using `hik.eval.Evaluator`.

```
python eval.py path/to/model data/
```

This will create a file `eval.pkl` that can be analyzed using Humans in Kitchens evaluation code.

## Reference

If you found this repository useful, please cite

```
@misc{mueller2024sast,
      title={Massively Multi-Person 3D Human Motion Forecasting with Scene Context}, 
      author={Felix B Mueller and Julian Tanke and Juergen Gall},
      year={2024},
      eprint={2409.12189},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.12189}, 
}
```