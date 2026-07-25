import pickle


def save_results(fname, results):
    """"""
    with open(fname, "wb") as f:
        pickle.dump(results, f)


def load_results(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
