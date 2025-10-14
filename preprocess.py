# scripts/preprocess.py

import argparse
import pandas as pd

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess raw dataset")
    p.add_argument("--input", required=True, help="Path to raw CSV file")
    p.add_argument("--output", required=True, help="Path to save cleaned CSV")
    return p


def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)
    df = df.drop_duplicates()
    df = df.fillna(0)
    df.to_csv(args.output, index=False)
    print(f"✅ Saved cleaned data to {args.output}")


if __name__ == "__main__":
    main()

