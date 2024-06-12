import yaml
import torch
import joblib


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("Value or filename cannot be None".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename)

    else:
        raise ValueError("Filename cannot be None".capitalize())


def config():
    with open("./config.yml", "r") as ymlfile:
        return yaml.safe_load(ymlfile)


def device_init(device="cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")
