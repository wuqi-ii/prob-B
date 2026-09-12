"""Run from any directory: python code/run.py --config configs/default.json."""
import argparse
from pathlib import Path
from bearing2.config import load_config
from bearing2.pipeline import run_pipeline


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="B题第二问：保留三项约束，在候选区域内优化代价")
    parser.add_argument("--config", type=Path, default=root / "configs/default.json")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    config = load_config(path)
    run_pipeline(root, config, plots=not args.skip_plots, sensitivity=not args.skip_sensitivity)


if __name__ == "__main__":
    main()
