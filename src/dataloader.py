import os
import shutil

# import torch
import pandas as pd

# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader

from typing import List

# class WordTorchSample:
#     """Word-associated information in torch format."""
#     def __init__(self, imagepath, value, transform=None):
#         """
#         Args:
#             imagepath (str): Path to the respective word-image
#             word (str): Transcription of the assciated word
#             transform (): Type of transformation applied on the image
#         """
#         self.transform = transform
#         self.imagepath


def strip_file(filename):
    """Strip #-comment lines from file represented by 'filename'

    Args:
        filename (str): source file to strip of #-comments
    """
    with open(filename, "rt") as f:
        lines = f.readlines()

    with open(filename, "wt") as f:
        for line in lines:
            if not line.startswith("#"):
                f.write(line)


class DatasetIAM:
    """Dataset specialisation for IAM data."""

    def __init__(self, words_file, filter_err=False, transform=None):
        """
        Args:
            words_file (str): Path to a modified ASCII file, i.e. path to "words.new"
            filter_err (bool): If set to True, retains only 'ok' word segmentation samples
            transform (): Any additional mapping for image transformations
        """

        self.words_file = words_file
        self.filter_err = filter_err
        self.transform = transform

        self.ref_length = 0

    def __len__(self):
        if self.ref_length == 0:
            with open(self.pathname, "rbU") as f:
                self.ref_length = sum(1 for _ in f)

        assert self.ref_length > 0
        return self.ref_length

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # TODO:
