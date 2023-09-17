import argparse

parser = argparse.ArgumentParser(description="select CTC decoding options")
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["best", "beam", "lexicon"],
    help="configure CTC decoding strategy",
)
parser.add_argument(
    "-v",
    "--verbosity",
    default=0,
    action="count",
    help="increase output verbosity",
)
args = parser.parse_args()

print(args.mode)
