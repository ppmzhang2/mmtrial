"""Environment variables and functions."""

from enum import Enum

import mmcv
import mmcv.utils


class TColor(Enum):
    """Colors for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_env() -> None:
    """Print the environment info."""
    for _, (k, v) in enumerate(mmcv.utils.collect_env().items()):
        print(f"{TColor.OKGREEN.value}=={k}== "  # noqa: T201
              f"{TColor.ENDC.value}{v}")


if __name__ == "__main__":
    print_env()
