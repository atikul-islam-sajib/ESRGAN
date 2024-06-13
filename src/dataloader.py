import os
import sys
import cv2
import zipfile
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("src/")

from utils import dump, load, config


class Loader:
    def __init__(self, image_path=None, image_size=64, split_size=0.20, batch_size=1):
        self.image_path = image_path
        self.image_size = image_size
        self.split_size = split_size
        self.batch_size = batch_size

        self.LR = []
        self.HR = []

        self.CONFIG = config()

    def split_dataset(self, X=None, y=None):
        if isinstance(X, list) and isinstance(y, list):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.split_size, random_state=42, shuffle=True
            )

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

    def transforms(self, type="lr"):
        if type == "lr":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        elif type == "hr":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size * 4, self.image_size * 4)),
                    transforms.ToTensor(),
                    transforms.CenterCrop((self.image_size * 4, self.image_size * 4)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def unzip_folder(self):
        if os.path.exists(self.CONFIG["path"]["RAW_DATA_PATH"]):
            with zipfile.ZipFile(self.image_path, "r") as zip_file:
                zip_file.extractall(os.path.join(self.CONFIG["path"]["RAW_DATA_PATH"]))
        else:
            raise Exception("RAW data path is not found".capitalize())

    def feature_extraction(self):

        self.directory = os.path.join(self.CONFIG["path"]["RAW_DATA_PATH"], "dataset")

        self.higher_resolution_images = os.path.join(self.directory, "HR")
        self.low_resolution_images = os.path.join(self.directory, "LR")

        for image in os.listdir(self.low_resolution_images):
            if image in os.listdir(self.higher_resolution_images):
                lower_resolution_image_path = os.path.join(
                    self.low_resolution_images, image
                )
                higher_resolution_image_path = os.path.join(
                    self.higher_resolution_images, image
                )

                lower_resolution_image = cv2.imread(lower_resolution_image_path)
                higher_resolution_image = cv2.imread(higher_resolution_image_path)

                lower_resolution_image = cv2.cvtColor(
                    lower_resolution_image, cv2.COLOR_BGR2RGB
                )
                higher_resolution_image = cv2.cvtColor(
                    higher_resolution_image, cv2.COLOR_BGR2RGB
                )

                lower_resolution_image = Image.fromarray(lower_resolution_image)
                higher_resolution_image = Image.fromarray(higher_resolution_image)

                self.LR.append(self.transforms(type="lr")(lower_resolution_image))
                self.HR.append(self.transforms(type="hr")(higher_resolution_image))

        assert len(self.LR) == len(self.HR)

        print("Total {} images have been captured".format(len(self.LR)).capitalize())

        return self.split_dataset(X=self.LR, y=self.HR)

    def create_dataloader(self):
        try:
            dataset = self.feature_extraction()

        except Exception as e:
            raise Exception("Feature extraction process has been failed".capitalize())

        else:
            train_dataloader = DataLoader(
                dataset=list(zip(dataset["X_train"], dataset["y_train"])),
                batch_size=self.batch_size,
                shuffle=True,
            )
            valid_dataloader = DataLoader(
                dataset=list(zip(dataset["X_test"], dataset["y_test"])),
                batch_size=self.batch_size * 8,
                shuffle=True,
            )

            for dataloader, filename in [
                (train_dataloader, "train_dataloader"),
                (valid_dataloader, "valid_dataloader"),
            ]:
                dump(
                    value=dataloader,
                    filename=os.path.join(
                        self.CONFIG["path"]["PROCESSED_DATA_PATH"], filename + ".pkl"
                    ),
                )

            print(
                "train and valid dataloader has been created in the folder : {}".format(
                    self.CONFIG["path"]["PROCESSED_DATA_PATH"]
                ).capitalize()
            )

    @staticmethod
    def plot_images():
        dataloader = load(
            filename=os.path.join(
                config()["path"]["PROCESSED_DATA_PATH"], "valid_dataloader.pkl"
            )
        )

        data, labels = next(iter(dataloader))

        plt.figure(figsize=(20, 10))

        for index, image in enumerate(data):
            X = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            y = labels[index].squeeze().permute(1, 2, 0).detach().cpu().numpy()

            X = (X - X.min()) / (X.max() - X.min())
            y = (y - y.min()) / (y.max() - y.min())

            plt.subplot(2 * 2, 2 * 4, 2 * index + 1)
            plt.imshow(X)
            plt.title("LR")
            plt.axis("off")

            plt.subplot(2 * 2, 2 * 4, 2 * index + 2)
            plt.imshow(y)
            plt.title("HR")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["ARTIFACTS_PATH"], "images.png"))
        plt.show()

        print(
            "Images have been saved in the folder : {}".format(
                config()["path"]["ARTIFACTS_PATH"]
            ).capitalize()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader for ESRGAN".title())
    parser.add_argument(
        "--image_path",
        type=str,
        default=config()["dataloader"]["image_path"],
        help="Define the dataset in zip format".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Define the image size".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config()["dataloader"]["split_size"],
        help="Define the split size".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config()["dataloader"]["batch_size"],
        help="Define the batch size".capitalize(),
    )
    args = parser.parse_args()

    loader = Loader(
        image_path=args.image_path,
        image_size=args.image_size,
        split_size=args.split_size,
        batch_size=args.batch_size,
    )

    loader.unzip_folder()
    loader.create_dataloader()

    Loader.plot_images()
