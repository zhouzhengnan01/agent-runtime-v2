from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DEIMv2 with torch_npu CUDA compatibility transfer enabled.")
    parser.add_argument("--train-py", required=True)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

    train_py = Path(args.train_py).resolve()
    sys.path.insert(0, str(train_py.parent))
    sys.argv = [str(train_py)] + list(args.train_args)
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
