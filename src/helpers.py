import os
import torch.nn as nn
import torch.optim as optim
from utils import config, load

from generator import Generator
from discriminator import Discriminator


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
        
        hello

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
    netD = Discriminator(in_channels=64, out_channels=64)

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
        
    criterion = 
