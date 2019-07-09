import argparse


def arg_common():

    parser = argparse.ArgumentParser(
        description="Argument for forgery detection")

    parser.add_argument(
        "--videoset", "-v", type=str, default="tmp_youtube", help="video dataset name"
    )
    parser.add_argument(
        "--root", type=str, default="~/dataset/video_forge/", help="root folder for dataset"
    )

    parser.add_argument("--test", action='store_true', help="test only mode")
    parser.add_argument("--validate", action='store_true',
                        help="whether to validate only mode")
    parser.add_argument("--with-src", action="store_true",
                        help="include source mask")

    parser.add_argument("--split", type=float, default=0.7, help="train split")

    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--clip", type=float, default=5,
                        help="gradient clipping")
    parser.add_argument("--model-type", default="deeplab", type=str,
                        help="model type (unet/albunet/deeplab)")
    parser.add_argument("--loss-type", default="bce", type=str,
                        help="loss type (bce / dice / l1)")
    parser.add_argument("--gamma", type=float, default=1,
                        help="gamma")

    parser.add_argument("--suffix", type=str, default="", help="model name suffix")
    parser.add_argument("--resume", type=int, default=1)

    return parser


def arg_main():
    parser = arg_common()
    parser.add_argument("--model", type=str, default="base", help="model name")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=20,
                        help="number of epoch")


    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")

    parser.add_argument("--boundary", action='store_true',
                        help="To include boundary for training")
    parser.add_argument("--gamma_b", type=float, default=1,
                        help="gamma for boundary loss")
    parser.add_argument("--size", type=int, default=224, help="image size")

    args = parser.parse_args()
    return args


def arg_main_search():
    parser = arg_common()

    parser.add_argument("--model", type=str, default="comp", help="model name")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=20000,
                        help="number of epoch")

    parser.add_argument("--batch-size", "-b", type=int,
                        default=20, help="batch size")
    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")
    parser.add_argument("--size", type=int, default=224, help="image size")

    parser.add_argument("--batch-size", "-b", type=int,
                        default=20, help="batch size")
    args = parser.parse_args()
    return args


def arg_main_track():
    parser = arg_common()
    parser.add_argument("--model", type=str,
                        default="track", help="model name")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of epoch")

    parser.add_argument("--batch-size", "-b", type=int,
                        default=20, help="batch size")
    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")
    parser.add_argument("--batch-size", "-b", type=int,
                        default=20, help="batch size")
    parser.add_argument("--size", type=int, default=224, help="image size")

    args = parser.parse_args()
    return args


def arg_main_tcn():
    parser = arg_common()
    parser.add_argument("--sep", type=int,
                        default=4, help="")
    parser.add_argument("--model", type=str,
                        default="tcn", help="model name")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of epoch")

    parser.add_argument("--batch-size", "-b", type=int,
                        default=20, help="batch size")
    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")\

    parser.add_argument("--level", type=int, default=2, help="level of tcn")
    parser.add_argument("--batch-size", "-b", type=int,
                        default=20, help="batch size")
    parser.add_argument("--size", type=int, default=224, help="image size")

    args = parser.parse_args()
    return args


def arg_main_match():
    parser = arg_common()
    parser.add_argument("--sep", type=int,
                        default=4, help="")
    parser.add_argument("--model", type=str,
                        default="match", help="model name")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of epoch")

    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--batch-size", "-b", type=int,
                        default=5, help="batch size")
    parser.add_argument("--size", type=int, default=320, help="image size")

    args = parser.parse_args()
    return args


def arg_main_im_match():
    parser = arg_common()
    parser.add_argument("--sep", type=int,
                        default=4, help="")
    parser.add_argument("--model", type=str,
                        default="immatch", help="model name")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument(
        "--test-path",
        type=str,
        default="./test_figs",
        help="path to save sample test figures",
    )
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of epoch")

    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--batch-size", "-b", type=int,
                        default=5, help="batch size")
    parser.add_argument("--size", type=int, default=320, help="image size")

    args = parser.parse_args()
    return args
