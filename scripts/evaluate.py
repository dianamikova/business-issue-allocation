import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_PATH, RESULTS_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = RESULTS_PATH if args.all_results else METRICS_PATH
    if not path.exists():
        raise FileNotFoundError("No metrics found. Run training first.")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
