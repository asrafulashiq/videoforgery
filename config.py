import argparse


def arg_common():

    parser = argparse.ArgumentParser(description="Argument for forgery detection")

    parser.add_argument(
        "--videoset", "-v", type=str, default="youtube", help="video dataset name"
    )
    parser.add_argument(
        "--root", type=str, default="~/dataset/video_forge/", help="root folder for dataset"
    )

    parser.add_argument("--test", action='store_true', help="test only mode")

    parser.add_argument("--size", type=int, default=224, help="image size")
    parser.add_argument("--split", type=float, default=0.8, help="train split")

    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--clip", type=float, default=1,
                        help="gradient clipping")
    return parser

def arg_main():
    parser = arg_common()
    parser.add_argument("--model", type=str, default="base2", help="model name")
    parser.add_argument("--ckpt", type=str, default=None, help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=20, help="number of epoch")

    parser.add_argument("--batch-size", "-b", type=int, default=20, help="batch size")
    parser.add_argument("--thres", type=float, default=0.5, help="threshold for detection")

    args = parser.parse_args()
    return args

def arg_main_search():
    parser = arg_common()
    parser.add_argument("--model", type=str, default="comp", help="model name")
    parser.add_argument("--ckpt", type=str, default=None, help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=20000, help="number of epoch")

    parser.add_argument("--batch-size", "-b", type=int, default=20, help="batch size")
    parser.add_argument("--thres", type=float, default=0.5, help="threshold for detection")

    args = parser.parse_args()
    return args


def arg_main_track():
    parser = arg_common()
    parser.add_argument("--model", type=str, default="track", help="model name")
    parser.add_argument("--ckpt", type=str, default=None, help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=100, help="number of epoch")

    parser.add_argument("--batch-size", "-b", type=int, default=20, help="batch size")
    parser.add_argument("--thres", type=float, default=0.5, help="threshold for detection")

    args = parser.parse_args()
    return args
