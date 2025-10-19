#!/bin/bash

python scripts/preprocess.py --train_path data/titanic/train.csv --test_path data/titanic/test.csv --output_train data/processed/train_processed.csv --output_test data/processed/test_processed.csv && \
python scripts/featurize.py --train_path data/processed/train_processed.csv --test_path data/processed/test_processed.csv --output_train data/featurized/train_features.csv --output_test data/featurized/test_features.csv --transformer_dir data/transformers --eval_dir data/eval --train_dir data/train --test_size 0.2 && \
python scripts/train.py --x_train_path data/train/X_train.csv --y_train_path data/train/y_train.csv --transformer_dir data/transformers --output_model models/random_forest_pipeline.pkl && \
python scripts/evaluate.py --model_path models/random_forest_pipeline.pkl --X_path data/eval/X_eval.csv --y_path data/eval/y_eval.csv --output_json reports/metrics.json && \
python scripts/predict.py --model_path models/random_forest_pipeline.pkl --features_path data/featurized/test_features.csv --output_path data/predictions/test_predictions.csv

