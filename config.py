import argparse

parser = argparse.ArgumentParser(description="Argument for forgery detection")

parser.add_argument("--model", type=str, default="base", help="model name")

parser.add_argument(
    "--videoset", "-v", type=str, default="SegTrackv2", help="video dataset name"
)

parser.add_argument(
    "--root", type=str, default="~/dataset/video_forge/", help="root folder for dataset"
)

parser.add_argument("--seed", type=float, default=0, help="random seed")
parser.add_argument("--batch-size", "-b", type=int, default=20, help="batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--epoch", type=int, default=20, help="number of epoch")

args = parser.parse_args()
print(args)
