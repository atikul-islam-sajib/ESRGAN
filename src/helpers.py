import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import config, load

from generator import Generator
from discriminator import Discriminator

from loss.VGG19 import VGG19
from loss.MSELoss import MSELoss


def load_dataset():

    if config()["path"]["PROCESSED_DATA_PATH"]:
        train_daloader = os.path.join(
            config()["path"]["PROCESSED_DATA_PATH"], "train_dataloader.pkl"
        )
        valid_dataloader = os.path.join(
            config()["path"]["PROCESSED_DATA_PATH"], "valid_dataloader.pkl"
        )

        return {
            "train_dataloader": load(train_daloader),
            "valid_dataloader": load(valid_dataloader),
        }

    else:
        raise Exception("No processed data found".capitalize())


def helper(**kwargs):
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]

    netG = Generator(in_channels=3, out_channels=64)
    netD = Discriminator(in_channels=3, out_channels=64)

    if adam:
        optimizerG = optim.Adam(params=netG.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerD = optim.Adam(params=netD.parameters(), lr=lr, betas=(beta1, beta2))

    elif SGD:
        optimizerG = optim.SGD(params=netG.parameters(), lr=lr, momentum=momentum)
        optimizerD = optim.SGD(params=netD.parameters(), lr=lr, momentum=momentum)

    try:
        dataset = load_dataset()

    except Exception as e:
        print(e)

    adversarial_loss = MSELoss(reduction="mean")
    perceptual_loss = VGG19(name="VGG19")

    return {
        "netG": netG,
        "netD": netD,
        "optimizerG": optimizerG,
        "optimizerD": optimizerD,
        "adversarial_loss": adversarial_loss,
        "perceptual_loss": perceptual_loss,
        "train_dataloader": dataset["train_dataloader"],
        "valid_dataloader": dataset["valid_dataloader"],
    }


if __name__ == "__main__":
    init = helper(
        lr=0.0002,
        adam=True,
        SGD=False,
        beta1=0.5,
        beta2=0.999,
        momentum=0.9,
    )

    assert init["netG"].__class__.__name__ == "Generator"
    assert init["netD"].__class__.__name__ == "Discriminator"
    assert init["optimizerG"].__class__.__name__ == "Adam"
    assert init["optimizerD"].__class__.__name__ == "Adam"
    assert init["adversarial_loss"].__class__.__name__ == "MSELoss"
    assert init["perceptual_loss"].__class__.__name__ == "VGG19"
    assert type(init["train_dataloader"]) == torch.utils.data.DataLoader
    assert type(init["valid_dataloader"]) == torch.utils.data.DataLoader
