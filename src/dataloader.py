class Loader:
    def __init__(self, image_path=None, image_size=64, split_size=0.20):
        self.image_path = image_path
        self.image_size = image_size
        self.split_size = split_size

    def split_dataset(self):
        pass

    def transforms(self):
        pass

    def unzip_folder(self):
        pass

    def feature_extraction(self):
        pass

    def create_dataloader(self):
        pass
