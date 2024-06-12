import torch
import torch.nn as nn
from tqdm import tqdm

from helpers import helper


class Trainer:
    def __init__(
        self,
        epochs=100,
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        adam=True,
        SGD=False,
        momentum=0.9,
        device="cuda",
        lr_scheduler=False,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam = adam
        self.SGD = SGD
        self.momentum = momentum
        self.device = device
        self.lr_scheduler = lr_scheduler

        self.init = helper(
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            adam=self.adam,
            SGD=self.SGD,
            momentum=self.momentum,
        )

        self.netG = self.init["netG"]
        self.netD = self.init["netD"]

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD = self.init["optimizerD"]

        self.adversarial_loss = self.init["adversarial_loss"]
        self.perceptual_loss = self.init["perceptual_loss"]

        self.train_dataloader = self.init["train_dataloader"]
        self.val_dataloader = self.init["valid_dataloader"]

    def train(self):
        pass


if __name__ == "__main__":
    trainer = Trainer(
        epochs=1,
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        adam=True,
        SGD=False,
        momentum=0.9,
        device="mps",
        lr_scheduler=False,
    )
