
import os
from sast.realism_classifier.data import load_dataset
from sast.realism_classifier.model import RealismClassifier

from torch.utils import data
from torchmetrics.classification import BinaryAUROC
import torch.nn.functional as F

from fire import Fire
from tqdm import tqdm

def auc(run_dir, data_path, device="cuda"):

    print("Loading validation data")

    val_gt_dl = data.DataLoader(
        load_dataset(f"{data_path}/gt_test.npz", device),
        batch_size=4096,
        shuffle=False,
    )

    val_models_dl = data.DataLoader(
        load_dataset(f"{data_path}/synthetic_val.npz", device),
        batch_size=4096,
        shuffle=False,
    )

    saves = sorted(os.listdir(run_dir))

    for filename in saves:
        if filename.startswith("save"):

            #print("Load ", filename)
            
            model = RealismClassifier.load(f"{run_dir}/{filename}", device=device)

            auc_combined = BinaryAUROC()
            auc_real = BinaryAUROC()
            auc_synthetic = BinaryAUROC()

            cross_entropy_real = 0
            cross_entropy_synthetic = 0
            n_samples_real = 0
            n_samples_synthetic = 0

            for X, y in val_gt_dl:
                y_pred = model(X)

                auc_combined.update(y_pred, y)
                auc_real.update(y_pred, y)
                cross_entropy_real += F.binary_cross_entropy(y_pred, y, reduction="sum").item()
                n_samples_real += len(y)

            for X, y in val_models_dl:
                y_pred = model(X)

                auc_combined.update(y_pred, y)
                auc_synthetic.update(y_pred, y)
                cross_entropy_synthetic += F.binary_cross_entropy(y_pred, y, reduction="sum").item()
                n_samples_synthetic += len(y)

            results = {
                "model": filename,
                "AUC/combined": auc_combined.compute().item(),
                "AUC/real": auc_real.compute().item(),
                "AUC/synthetic": auc_synthetic.compute().item(),
                "CrossEntropy/real": cross_entropy_real / n_samples_real,
                "CrossEntropy/synthetic": cross_entropy_synthetic / n_samples_synthetic,
                "Len/real": n_samples_real,
                "Len/synthetic": n_samples_synthetic,
            }

            print(results)


if __name__ == "__main__":
    Fire({"auc": auc})
