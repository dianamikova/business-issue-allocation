import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = Predictor()
    result = predictor.predict(args.text)
    print({"text": args.text, **result})


if __name__ == "__main__":
    main()
