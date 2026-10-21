"""Command line interface for labelmetoyolo."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .converter import TASKS, convert


def _parse_classes(values: Optional[List[str]]) -> Optional[List[str]]:
    """Accept class names directly, or a single .txt file with one per line."""
    if values and len(values) == 1 and Path(values[0]).is_file():
        lines = Path(values[0]).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]
    return values


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="labelmetoyolo",
        description="Convert LabelMe JSON annotations to a YOLO dataset.",
    )
    parser.add_argument("input", help="folder containing LabelMe .json files")
    parser.add_argument("-o", "--output", default="yolo_dataset", help="output folder (default: yolo_dataset)")
    parser.add_argument("-t", "--task", default="detect", choices=TASKS, help="label format (default: detect)")
    parser.add_argument(
        "-s", "--split", nargs="+", type=float, metavar="RATIO",
        help="train/val or train/val/test ratios, e.g. --split 0.8 0.2 or --split 70 20 10",
    )
    parser.add_argument(
        "-c", "--classes", nargs="+", metavar="NAME",
        help="class names in id order, or a .txt file with one name per line",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split (default: 42)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        convert(
            args.input,
            args.output,
            task=args.task,
            classes=_parse_classes(args.classes),
            split=args.split,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
