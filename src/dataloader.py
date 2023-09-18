import os
import cv2
import csv
import torch
import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Callable

from src.data_augmentation import DataAugPolicy


class WordSample:
    """Word-associated information in torch format."""

    def __init__(self, imagepath: str, value: str, bin_threshold: int):
        """
        Args:
            imagepath: Path to the image of the associated word
            word: Transcription of the associated word
            bin_threshold: Threshold for binarization of grayscale image
        """
        self.imagepath = imagepath
        self.image = cv2.imread(self.imagepath, cv2.IMREAD_GRAYSCALE)
        self.value = value
        self.bin_threshold = bin_threshold

    # TODO: check whether adaptive thresholding works better
    def apply_global_binarization(self):
        self.image = cv2.threshold(
            self.image, self.bin_threshold, 255, cv2.THRESH_BINARY
        )


class DatasetIAM(Dataset):
    """Dataset specialisation for IAM data."""

    def __init__(
        self,
        words_image_path: str,
        words_meta_path: str,
        filter_err: bool = False,
        preprocess: Optional[Callable] = None,
        data_augmentation: Optional[DataAugPolicy] = False,
    ):
        """
        Args:
            words_image_dir: Path to directory containing the word images
            words_meta_path: Path to a (modified) ASCII file, i.e. path to "words.new"
            filter_err: If set to True, retains only 'ok' word segmentation samples
            preprocess: Any additional mapping for image transformations
        """

        self.words_image_path = words_image_path
        self.words_meta_path = words_meta_path

        self.words = pd.read_csv(
            self.words_meta_path,
            delimiter=" ",
            header=None,  # take care not to lose the first row
            skip_blank_lines=True,  # include only valid lines
            quoting=csv.QUOTE_NONE,  # manage words that are quotations marks
        )

        # Filter 'err' transcriptions with pd functionality
        if filter_err:
            self.words.drop(
                self.words[self.words.iloc[:, 1] != "ok"].index, inplace=True
            )

        # TODO: preprocess and data augmentation
        # Idea: implement preprocess as methods of WordSample class
        self.preprocess = preprocess
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.words.iloc[idx, 0]
        author, text, _, _ = tuple(image_name.split("-"))

        image_path = os.path.join(
            self.words_image_path, f"{author}", f"{author}-{text}", f"{image_name}.png"
        )

        # image = Image.open(image_path)
        # if not torch.is_tensor(image):
        #     image = transforms.functional.pil_to_tensor(
        #         image
        #     )  # assuming image is PIL.Image

        word_sample = WordSample(
            imagepath=image_path,
            value=self.words.iloc[idx, -1],
            bin_threshold=self.words.iloc[idx, -2],
        )

        return word_sample


# data_prefix = "/home/nasko/ctc-handwriting/data"
# ds = DatasetIAM(
#     words_image_path=f"{data_prefix}/words-images/",
#     words_meta_path=f"{data_prefix}/words-meta/words.new",
# )
# for i in range(10):
#     print(ds[i].image.shape)
