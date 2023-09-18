import os
import csv
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class WordTorchSample:
    """Word-associated information in torch format."""

    def __init__(self, image, value):
        """
        Args:
            image (torch.Tensor): Tensorised image corresponding to word
            word (str): Transcription of the assciated word
        """
        self.image = image
        self.value = value


class DatasetIAM(Dataset):
    """Dataset specialisation for IAM data."""

    def __init__(
        self, words_image_path, words_meta_path, filter_err=False, transform=None
    ):
        """
        Args:
            words_image_dir (str): Path to directory containing the word images
            words_meta_path (str): Path to a (modified) ASCII file, i.e. path to "words.new"
            filter_err (bool): If set to True, retains only 'ok' word segmentation samples
            transform (): Any additional mapping for image transformations
        """

        self.words_image_path = words_image_path
        self.words_meta_path = words_meta_path

        self.words = pd.read_csv(
            self.words_meta_path,
            delimiter=" ",
            header=None,  # take care not to lose the first row
            quoting=csv.QUOTE_NONE,  # manage words that are quotations marks
        )

        # Filter 'err' transcriptions with pd functionality
        if filter_err:
            self.words.drop(
                self.words[self.words.iloc[:, 1] != "ok"].index, inplace=True
            )

        # TODO: transforms to boost grayscale, etc. Also, think about data augmentation...
        self.transform = transform

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # First, fetch image as png file
        image_name = self.words.iloc[idx, 0]
        author, text, _, _ = tuple(image_name.split("-"))

        image_path = self.words_image_path
        image_path = os.path.join(
            image_path, f"{author}", f"{author}-{text}", f"{image_name}.png"
        )

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        # Check agains self.transform
        if not torch.is_tensor(image):
            image = transforms.functional.pil_to_tensor(
                image
            )  # assuming image is PIL.Image

        word_sample = WordTorchSample(
            image=image,
            value=self.words.iloc[idx, -1],
        )

        return word_sample


data_prefix = "/home/nasko/ctc-handwriting/data"
ds = DatasetIAM(
    words_image_path=f"{data_prefix}/words-images/",
    words_meta_path=f"{data_prefix}/words-meta/words.new",
)
ds[1]
