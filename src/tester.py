import os
import sys
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append("/src/")

from generator import Generator
from utils import config, load, device_init


class Tester:
    def __init__(self, model="best", dataloader="valid", device="cuda"):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.device = device_init(device=self.device)

        self.netG = Generator(in_channels=3, out_channels=64).to(self.device)

    def select_best_model(self):
        if self.model == "best":
            state_dict = torch.load(
                os.path.join(config()["path"]["BEST_MODEL_CHECKPOINT_PATH"], "netG.pth")
            )

            return state_dict["netG"]

        else:
            return torch.load(self.model)

    def load_dataloader(self):
        if self.dataloader == "valid":
            return load(
                filename=os.path.join(
                    config()["path"]["PROCESSED_DATA_PATH"], "valid_dataloader.pkl"
                )
            )

        else:
            return load(
                filename=os.path.join(
                    config()["path"]["PROCESSED_DATA_PATH"], "train_dataloader.pkl"
                )
            )

    def plot(self, X=None, y=None, generated_hr_image=None):
        plt.figure(figsize=(40, 15))

        for index, image in enumerate(generated_hr_image):
            gen_image = image.permute(1, 2, 0).detach().cpu().numpy()
            real_lr_image = X[index].permute(1, 2, 0).detach().cpu().numpy()
            real_hr_image = y[index].permute(1, 2, 0).detach().cpu().numpy()

            gen_image = (gen_image - gen_image.min()) / (
                gen_image.max() - gen_image.min()
            )
            real_lr_image = (real_lr_image - real_lr_image.min()) / (
                real_lr_image.max() - real_lr_image.min()
            )
            real_hr_image = (real_hr_image - real_hr_image.min()) / (
                real_hr_image.max() - real_hr_image.min()
            )

            plt.subplot(3 * 2, 3 * 4, 3 * index + 2)
            plt.imshow(real_lr_image)
            plt.title("real_LR".capitalize())
            plt.axis("off")

            plt.subplot(3 * 2, 3 * 4, 3 * index + 1)
            plt.imshow(gen_image)
            plt.title("Generated".capitalize())
            plt.axis("off")

            plt.subplot(3 * 2, 3 * 4, 3 * index + 3)
            plt.imshow(real_hr_image)
            plt.title("real_HR".capitalize())
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["TEST_IMAGE_PATH"], "test.png"))
        plt.show()

        print("Image is saved in the folder: ", config()["path"]["TEST_IMAGE_PATH"])

    def test(self):
        try:

            self.netG.load_state_dict(self.select_best_model())

        except Exception as e:
            print("0000", e)
            return

        try:

            datloader = self.load_dataloader()

            X, y = next(iter(datloader))

        except Exception as e:
            print(e)
            return

        generated_hr_image = self.netG(X.to(self.device))

        try:
            self.plot(X, y, generated_hr_image)

        except Exception as e:
            print(e)
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model for ESRGAN".title())
    parser.add_argument(
        "--model",
        type=str,
        default=config()["tester"]["model"],
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default=config()["tester"]["dataloader"],
        help="Path to the dataloader pickle file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["tester"]["device"],
        help="Device to run the model",
    )

    args = parser.parse_args()

    tester = Tester(
        model=args.model,
        dataloader=args.dataloader,
        device=args.device,
    )

    tester.test()
