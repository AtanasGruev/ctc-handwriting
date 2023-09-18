# import argparse

# parser = argparse.ArgumentParser(description="select CTC decoding options")
# parser.add_argument(
#     "-m",
#     "--mode",
#     type=str,
#     choices=["best", "beam", "lexicon"],
#     help="configure CTC decoding strategy",
# )
# parser.add_argument(
#     "-v",
#     "--verbosity",
#     default=0,
#     action="count",
#     help="increase output verbosity",
# )
# args = parser.parse_args()

# print(args.mode)

import cv2
from matplotlib import pyplot as plt
from src.dataloader import DatasetIAM

data_prefix = "/home/nasko/ctc-handwriting/data"
ds = DatasetIAM(
    words_image_path=f"{data_prefix}/words-images/",
    words_meta_path=f"{data_prefix}/words-meta/words.new",
)
img = cv2.imread(ds[1].imagepath)
plt.figure(figsize=(8, 8))
plt.imshow(img)
