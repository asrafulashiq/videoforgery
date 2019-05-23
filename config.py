import argparse

parser = argparse.ArgumentParser(description="Argument for forgery detection")

parser.add_argument(
    "--videoset", "-v", type=str, default="SegTrackv2", help="video dataset name"
)

parser.add_argument(
    "--root", type=str, default="~/dataset/video_forge/", help="root folder for dataset"
)

parser.add_argument("--seed", type=float, default=0, help="random seed")

args = parser.parse_args()
print(args)
