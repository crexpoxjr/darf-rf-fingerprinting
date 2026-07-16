#!/usr/bin/env python3
"""Compatibility entry point for ORACLE CNN training.

This preserves `python train_oracle_cnn.py` while delegating to the
configuration-driven ORACLE training pipeline in `src.training.train`.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import main as train_main


def main() -> None:
    if len(sys.argv) == 1:
        default_config = PROJECT_ROOT / "configs" / "oracle_cnn.yaml"
        sys.argv.extend(["--config", str(default_config)])

    train_main()


if __name__ == "__main__":
    main()
