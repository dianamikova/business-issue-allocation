import csv
import json
from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_dataframe(df: pd.DataFrame, text_column: str, target_column: str, id_column: str) -> None:
    required_columns = {id_column, text_column, target_column}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[text_column].isna().any():
        raise ValueError("Dataset contains null text values.")

    if df[target_column].isna().any():
        raise ValueError("Dataset contains null label values.")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_csv(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as src:
        with destination.open("w", encoding="utf-8", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            writer.writerows(reader)
