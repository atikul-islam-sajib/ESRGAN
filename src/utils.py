import yaml
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
