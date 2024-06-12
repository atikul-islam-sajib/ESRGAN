import os
import torch
import mlflow
import argparse
import numpy as np
from tqdm import tqdm
from helpers import helper
import matplotlib.pyplot as plt
from dagshub import dagshub_logger
from torchvision.utils import save_image
from utils import config, load, dump, device_init, weight_init


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
        content_loss=0.01,
        pixel_loss=0.05,
        device="cuda",
        lr_scheduler=False,
        is_weight_init=False,
        verbose=True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam = adam
        self.SGD = SGD
        self.momentum = momentum
        self.content_loss = content_loss
        self.pixel_loss = pixel_loss
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.is_weight_init = is_weight_init
        self.is_verbose = verbose

        self.init = helper(
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            adam=self.adam,
            SGD=self.SGD,
            momentum=self.momentum,
        )

        self.device = device_init(device=self.device)

        self.netG = self.init["netG"].to(self.device)
        self.netD = self.init["netD"].to(self.device)

        if self.is_weight_init:
            self.netG.apply(weight_init)
            self.netD.apply(weight_init)

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD = self.init["optimizerD"]

        self.adversarial_loss = self.init["adversarial_loss"].to(self.device)
        self.perceptual_loss = self.init["perceptual_loss"].to(self.device)

        self.train_dataloader = self.init["train_dataloader"]
        self.val_dataloader = self.init["valid_dataloader"]

        self.loss = float("inf")

        self.CONFIG = config()

        self.history = {"netG_loss": [], "netD_loss": []}

        os.environ["MLFLOW_TRACKING_URI"] = (
            "https://dagshub.com/atikul-islam-sajib/ESRGAN.mlflow"
        )

        # Authenticate with your DagsHub token
        os.environ["MLFLOW_TRACKING_USERNAME"] = "atikul-islam-sajib"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = (
            "0c9062f98631dcdb3a9b2d56ebba7ff850a19139"
        )

        mlflow.set_experiment(experiment_name="ESRGAN".title())

    def l1_loss(self, model=None):
        if model is not None:
            return sum(torch.norm(params, 1) for params in model.parameters())

    def l2_loss(self, model=None):
        if model is not None:
            return sum(torch.norm(params, 2) for params in model.parameters())

    def elastic_loss(self, model=None):
        l1 = self.l1_loss(model)
        l2 = self.l2_loss(model)

        return l1 + l2

    def update_netG(self, **kwargs):
        self.optimizerG.zero_grad()

        lr_image = kwargs["lr_image"]
        hr_image = kwargs["hr_image"]

        generated_hr_image = self.netG(lr_image)
        predicted_generated_hr_image = self.netD(generated_hr_image)

        real_features = self.perceptual_loss(hr_image)
        generated_features = self.perceptual_loss(generated_hr_image)

        content_loss = torch.abs(real_features - generated_features).mean()
        pixelwise_loss = torch.abs(hr_image - generated_hr_image).mean()

        loss_generated_hr_image = self.adversarial_loss(
            predicted_generated_hr_image, torch.ones_like(predicted_generated_hr_image)
        )

        total_loss = (
            loss_generated_hr_image
            + self.content_loss * content_loss
            + self.pixel_loss * pixelwise_loss
        )

        total_loss.backward()
        self.optimizerG.step()

        return total_loss.item()

    def update_netD(self, **kwargs):
        self.optimizerD.zero_grad()

        lr_image = kwargs["lr_image"]
        hr_image = kwargs["hr_image"]

        generated_hr_image = self.netG(lr_image)
        predicted_generated_hr_image = self.netD(generated_hr_image)

        predicted_real_hr_image = self.netD(hr_image)

        loss_generated_hr_image = self.adversarial_loss(
            predicted_generated_hr_image, torch.zeros_like(predicted_generated_hr_image)
        )
        loss_real_hr_image = self.adversarial_loss(
            predicted_real_hr_image, torch.ones_like(predicted_real_hr_image)
        )

        total_loss = 0.5 * (loss_generated_hr_image + loss_real_hr_image)

        total_loss.backward()
        self.optimizerD.step()

        return total_loss.item()

    def saved_checkpoints(self, **kwargs):
        netG_loss = kwargs["netG_loss"]

        if self.loss > netG_loss:
            self.loss = netG_loss

            torch.save(
                {
                    "netG": self.netG.state_dict(),
                    "loss": netG_loss,
                    "epoch": kwargs["epoch"],
                },
                os.path.join(
                    self.CONFIG["path"]["BEST_MODEL_CHECKPOINT_PATH"], "netG.pth"
                ),
            )
        torch.save(
            self.netG.state_dict(),
            os.path.join(
                self.CONFIG["path"]["TRAIN_MODEL_CHECKPOINT_PATH"],
                "netG{}.pth".format(kwargs["epoch"] + 1),
            ),
        )

    def show_progress(self, **kwargs):
        if self.is_verbose:
            print(
                "Epochs - [{}/{}] - netG_loss: [{:.4f}] - netD_loss: [{:.4f}]".format(
                    kwargs["epoch"] + 1,
                    self.epochs,
                    kwargs["netG_loss"],
                    kwargs["netD_loss"],
                )
            )

        else:
            print(
                "Epochs - [{}/{}] is completed".capitalize().format(
                    kwargs["epoch"] + 1, self.epochs
                )
            )

    def saved_train_images(self, epoch=None):

        lr_image, hr_image = next(iter(self.train_dataloader))
        lr_image = lr_image.to(self.device)
        hr_image = hr_image.to(self.device)

        generated_hr_image = self.netG(lr_image)

        save_image(
            generated_hr_image,
            os.path.join(
                config()["path"]["TRAIN_IMAGES_PATH"],
                "image_{}.png".format(epoch + 1),
            ),
        )

    def train(self):
        with mlflow.start_run() as run:
            for epoch in tqdm(range(self.epochs)):
                self.netD_loss = []
                self.netG_loss = []

                for _, (lr_image, hr_image) in enumerate(self.train_dataloader):

                    lr_image = lr_image.to(self.device)
                    hr_image = hr_image.to(self.device)

                    self.netD_loss.append(
                        self.update_netD(lr_image=lr_image, hr_image=hr_image)
                    )
                    self.netG_loss.append(
                        self.update_netG(lr_image=lr_image, hr_image=hr_image)
                    )

                self.show_progress(
                    epoch=epoch,
                    netG_loss=np.mean(self.netG_loss),
                    netD_loss=np.mean(self.netD_loss),
                )

                self.saved_train_images(epoch=epoch)
                self.saved_checkpoints(netG_loss=np.mean(self.netG_loss), epoch=epoch)

                self.history["netG_loss"].append(np.mean(self.netG_loss))
                self.history["netD_loss"].append(np.mean(self.netD_loss))

                mlflow.log_metric("netG_loss", np.mean(self.netG_loss), step=epoch + 1)
                mlflow.log_metric("netD_loss", np.mean(self.netD_loss), step=epoch + 1)

            dump(
                value=self.history,
                filename=os.path.join(
                    self.CONFIG["path"]["METRICS_PATH"], "history.pkl"
                ),
            )

            mlflow.log_params(
                {
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "beta1": self.beta1,
                    "beta2": self.beta2,
                    "momentum": self.momentum,
                    "adam": self.adam,
                    "SGD": self.SGD,
                    "content_loss": self.content_loss,
                    "pixel_loss": self.pixel_loss,
                    "adversarial_loss": self.adversarial_loss,
                    "adversarial_loss": self.adversarial_loss.__class__.__name__,
                    "lr_scheduler": self.lr_scheduler,
                    "weight_init": self.is_weight_init,
                    "verbose": self.is_verbose,
                    "device": self.device,
                }
            )

            mlflow.pytorch.log_model(self.netG, "model")

    @staticmethod
    def plot_history():

        if os.path.exists(config()["path"]["METRICS_PATH"]):
            history = load(
                filename=os.path.join(config()["path"]["METRICS_PATH"], "history.pkl")
            )

        else:
            raise Exception("Metrics path cannot be extraced".capitalize())

        plt.figure(figsize=(10, 5))

        plt.plot(history["netG_loss"], label="netG_loss")
        plt.plot(history["netD_loss"], label="netD_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for ESRGAN".title())
    parser.add_argument(
        "--epochs",
        type=int,
        default=config()["trainer"]["epochs"],
        help="Define the epochs for training the model".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["trainer"]["lr"],
        help="Define the learning rate for training the model".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["trainer"]["beta1"],
        help="Define the beta1 for training the model".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["trainer"]["beta2"],
        help="Define the beta2 for training the model".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["trainer"]["adam"],
        help="Define if the optimizer is adam".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["trainer"]["SGD"],
        help="Define if the optimizer is SGD".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config()["trainer"]["momentum"],
        help="Define the momentum for training the model".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Define the device for training the model".capitalize(),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=config()["trainer"]["lr_scheduler"],
        help="Define if the learning rate scheduler is used".capitalize(),
    )
    parser.add_argument(
        "--weight_init",
        type=bool,
        default=config()["trainer"]["weight_init"],
        help="Define if the weight initialization is used".capitalize(),
    )
    parser.add_argument(
        "--content_loss",
        type=float,
        default=config()["trainer"]["content_loss"],
        help="Define the content loss for training the model".capitalize(),
    )
    parser.add_argument(
        "--pixel_loss",
        type=float,
        default=config()["trainer"]["pixel_loss"],
        help="Define the pixel loss for training the model".capitalize(),
    )

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        adam=args.adam,
        SGD=args.SGD,
        momentum=args.momentum,
        content_loss=args.content_loss,
        pixel_loss=args.pixel_loss,
        device=args.device,
        lr_scheduler=args.lr_scheduler,
        is_weight_init=args.weight_init,
    )

    trainer.train()
    trainer.plot_history()
