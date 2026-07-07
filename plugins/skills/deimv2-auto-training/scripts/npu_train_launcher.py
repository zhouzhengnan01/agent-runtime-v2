from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DEIMv2 with torch_npu CUDA compatibility transfer enabled.")
    parser.add_argument("--train-py", required=True)
    # 其余参数全部原样转发给 DEIMv2 的 train.py。这里使用 parse_known_args，
    # 避免 -c、-d、-t 等 train.py 参数被 launcher 自己的 argparse 拦截。
    args, train_args = parser.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

    train_py = Path(args.train_py).resolve()
    sys.path.insert(0, str(train_py.parent))
    sys.argv = [str(train_py)] + list(train_args)
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
