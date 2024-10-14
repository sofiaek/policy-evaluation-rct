import argparse
import os
import datetime
import json
import subprocess
import logging
import sys


def generate_folder_path(base_folder, exp_folder_name):
    """Generate a folder path with the current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
    return os.path.join(os.getcwd(), base_folder, exp_folder_name, timestamp)


def set_out_dir(args):
    folder = generate_folder_path("results", args.exp_folder_name)
    os.makedirs(folder, exist_ok=True)
    return folder


def get_full_path(folder, name):
    return os.path.join(folder, name)


def get_git_diff() -> str:
    diff = (
        subprocess.check_output(["git", "diff", "HEAD"])
        .decode(sys.stdout.encoding)
        .strip()
    )
    if diff:
        message = "It is uncommited changes in the code."
        logging.warning(message)
        print("WARNING: " + message)
    return diff


def save_hyperparams(folder, args):
    with open(get_full_path(folder, "config.json"), "w") as handle:
        json.dump(vars(args), handle, indent="\t")


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def save_git_hash(folder):
    with open(get_full_path(folder, "git.txt"), "w", encoding="utf-8") as handle:
        handle.write(get_git_revision_hash() + "\n")
        handle.write(get_git_diff())


def save_logging(folder, args):
    handlers = [
        logging.FileHandler(get_full_path(folder, "debug.log")),
    ]
    if args.print_logs:
        handlers += [logging.StreamHandler()]
    if args.debug:
        logging.basicConfig(handlers=handlers, encoding="utf-8", level=logging.DEBUG)
    else:
        logging.basicConfig(handlers=handlers, encoding="utf-8", level=logging.INFO)

    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)


def save_logging_and_setup(args):
    # create output directory
    out_dir = set_out_dir(args)

    # save experiment details
    save_logging(out_dir, args)
    save_hyperparams(out_dir, args)
    save_git_hash(out_dir)

    return out_dir


def add_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("Save utils")

    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="logging in debug mode",
    )
    parser.add_argument(
        "--print_logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print logging",
    )
    parser.add_argument(
        "--exp_folder_name",
        default="exp",
        help="name of experiment folder (default: %(default)s)",
    )
    return parent_parser
