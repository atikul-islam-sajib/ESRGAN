import argparse
from utils import config
from tester import Tester
from trainer import Trainer
from dataloader import Loader


def cli():
    parser = argparse.ArgumentParser(description="CLI for ESRGAN".title())
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
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:

        loader = Loader(
            image_path=args.image_path,
            image_size=args.image_size,
            split_size=args.split_size,
            batch_size=args.batch_size,
        )

        loader.unzip_folder()
        loader.create_dataloader()

        Loader.plot_images()

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

    if args.test:

        tester = Tester(
            model=args.model,
            dataloader=args.dataloader,
            device=args.device,
        )

        tester.test()


if __name__ == "__main__":
    cli()
