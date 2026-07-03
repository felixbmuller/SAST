import pickle
from fire import Fire
import torch

import torch.utils.data as data
import numpy as np


def process_eval_pkl(eval_file, gt=False):
    with open(eval_file, "rb") as fp:
        eval = pickle.load(fp)

    out = []

    for cat, item_list in eval.items():
        for idx, item in enumerate(item_list):
            seq = item["seq_out_pred"] if not gt else item["seq_out_gt"]

            mask = item["masks_out"].astype("bool")

            # remove dummy batch dimension if present
            if len(seq.shape) == 5:
                assert seq.shape[0] == 1, str(seq.shape)
                seq = seq[0]

            for iperson in range(0, seq.shape[0]):
                out.append((seq[iperson], mask[iperson]))

    return out


def assemble_dataset(synthetic, real, length, stepsize, path):
    print(f"Assembling {path}")
    print(f"  Got {len(synthetic)=}, {len(real)=}")

    skipped = 0

    synthetic_words = []

    for seq, mask in synthetic:
        for iframe in range(0, seq.shape[0] - length + 1, stepsize):
            if not mask[iframe : iframe + length].all():
                # skip partially masked motion words
                skipped += 1
                continue

            synthetic_words.append(seq[iframe : iframe + length])

    print(f"  Skipped {skipped} synthetic samples because of masking")

    skipped = 0

    real_words = []

    for seq, mask in real:
        for iframe in range(0, seq.shape[0] - length + 1, stepsize):
            if not mask[iframe : iframe + length].all():
                # skip partially masked motion words
                skipped += 1
                continue

            real_words.append(seq[iframe : iframe + length])

    print(f"  Skipped {skipped} real samples because of masking")

    print(f"  Produces {len(synthetic_words)=}, {len(real_words)=}")

    print(
        f"  Proportion of real samples: {len(real_words)/(len(synthetic_words)+len(real_words))}"
    )

    X = np.stack(synthetic_words + real_words).astype(
        "float32"
    )  # batch, frame, joint, dim

    y = np.ones(X.shape[0], dtype=np.float32)
    y[: len(synthetic_words)] = 0.0

    print(f"SHAPES before shuffle {X.shape=}, {y.shape=}")

    gen = np.random.default_rng(42)
    idx = np.arange(len(y))
    gen.shuffle(idx)

    print(f"Shape idx shuffle {idx.shape=}")

    X = X[idx]
    y = y[idx]

    print(f"SHAPES after shuffle {X.shape=}, {y.shape=}")

    np.savez(path, X=X, y=y)


def prepare_datasets(data_dir, length=50, stepsize=5):
    """Binary Labels:
    0: synthetic
    1: real

    Parameters
    ----------
    data_dir : _type_
        _description_
    length : int, optional
        _description_, by default 50
    stepsize : int, optional
        _description_, by default 5
    """
    print(torch.__version__)

    files = [
        "hisrep_D.pkl",
        "mlp_D.pkl",
        "Ours_D.pkl",
        "mrt_D.pkl",
        "tripod_D.pkl",
    ]

    gen = torch.Generator().manual_seed(42)

    synthetic_samples_train = []

    synthetic_test = {}

    for file in files:
        out = process_eval_pkl(f"{data_dir}/{file}")

        train, test = data.random_split(out, [0.16, 0.84], generator=gen)

        synthetic_samples_train += train

        assemble_dataset(
            synthetic=test,
            real=[],
            length=length,
            stepsize=stepsize,
            path=f"{data_dir}/{file.split('.')[0]}_test",
        )

        synthetic_test[file] = test

    real_motion = process_eval_pkl(f"{data_dir}/{files[0]}", gt=True)

    real_train, real_test = data.random_split(real_motion, [0.8, 0.2], generator=gen)

    assemble_dataset(
        synthetic=synthetic_samples_train,
        real=real_train,
        length=length,
        stepsize=stepsize,
        path=f"{data_dir}/realism_train",
    )

    assemble_dataset(
        synthetic=[],
        real=real_test,
        length=length,
        stepsize=stepsize,
        path=f"{data_dir}/gt_test",
    )


def synthetic_whole_sequences(data_dir):
    """Binary Labels:
    0: synthetic
    1: real

    Parameters
    ----------
    data_dir : _type_
        _description_
    length : int, optional
        _description_, by default 50
    stepsize : int, optional
        _description_, by default 5
    """
    print(torch.__version__)

    import os

    base_dir = os.path.expanduser('~/media/2023 Anticipating Social Interactions in Environments/models/')

    files = {
        'NormalScaling': f'{base_dir}/M6_Normal/model_step=800000_epoch=284_df5bad_reproduce_out250clip_sample=False.pkl',
        'NonCausal': f'{base_dir}/M6_Normal/model_step=800000_epoch=284_df5bad_reproduce_out250clip_sample=False.pkl',
        'NoScene': f'{base_dir}/T03_NoScene/model_step=0680000_epoch=238_4b0a27_longer_clip_range=3_noisy_in_seq=False.pkl',
        'NoOthers': f'{base_dir}/T02_NoOthers/model_step=0680000_epoch=238_4b0a27_longer_clip_range=3_noisy_in_seq=False.pkl',
    }


    gen = torch.Generator().manual_seed(42)

    for name, file in files.items():
        out = process_eval_pkl(f"{file}")

        train, test = data.random_split(out, [0.16, 0.84], generator=gen)

        torch.save(test, f"{data_dir}/{name}_test_seq.pkl")

    #real_motion = process_eval_pkl(f"{data_dir}/{files[0]}", gt=True)

    #real_train, real_test = data.random_split(real_motion, [0.8, 0.2], generator=gen)

    #torch.save(real_test, f"{data_dir}/gt_test_seq.pkl")


def create_val_synthetic(data_dir):
    test_files = [
        "mlp_D_test.npz",
        "mrt_D_test.npz",
        "Ours_D_test.npz",
        "tripod_D_test.npz",
    ]

    Xs = []
    ys = []

    for file in test_files:
        print("File ", file)
        npz = np.load(f"{data_dir}/{file}")
        size = int(npz["y"].shape[0] * 0.1)
        print("  Select first samples ", size)

        Xs.append(npz["X"][:size])
        ys.append(npz["y"][:size])

    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    print(f"Val_synthetic: {X.shape=}, {y.shape=}")

    np.savez(f"{data_dir}/synthetic_val.npz", X=X, y=y)


def load_dataset(path, device="cpu"):
    npz = np.load(path)
    # print(f"Loaded dims {npz['X'].shape=}, {npz['y'].shape=}")
    X = torch.from_numpy(npz["X"]).to(device)
    y = torch.from_numpy(npz["y"]).to(device)

    return data.TensorDataset(X, y)


if __name__ == "__main__":
    Fire(
        {
            "prepare_datasets": prepare_datasets,
            "create_val_synthetic": create_val_synthetic,
            "synthetic_whole_sequences": synthetic_whole_sequences,
        }
    )
