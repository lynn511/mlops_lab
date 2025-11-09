"""
Script entry point for data preprocessing.
Uses mlops_lab.preprocessing.preprocessor functions.
"""

"""
Script entry point for data preprocessing.
Uses mlops_lab.preprocessing.preprocessor functions.
"""

import argparse
import warnings
from pathlib import Path
from mlops_lab.preprocessing.preprocessor import load_data, clean_data, split_data



warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_train", type=str, required=True)
    parser.add_argument("--output_test", type=str, required=True)

    args = parser.parse_args()

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train, test = load_data(args.train_path, args.test_path)

    print("Cleaning data...")
    df = clean_data(train, test)

    print("Splitting data...")
    train_processed, test_processed = split_data(df)

    print("Saving preprocessed data...")
    train_processed.to_csv(args.output_train, index=False)
    test_processed.to_csv(args.output_test, index=False)

    print("Done ✅")


if __name__ == "__main__":
    main()
