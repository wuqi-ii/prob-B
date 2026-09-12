"""Rebuild plots from saved solver data, without rerunning optimization."""
import argparse
from pathlib import Path
from bearing2.plotting import plot_saved_results


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="重绘候选区域、最终选点、代价及敏感性图")
    parser.add_argument("--name", default="default")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Invalid run name.")
    plot_saved_results(root, args.name)
