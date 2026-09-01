"""Fine-tune a Czech transformer encoder"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CLEAN_DATA_PATH,
    PRIMARY_METRIC,
    ROOT,
    SEED,
    TRANSFORMER_EARLY_STOPPING_PATIENCE,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_LR,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_WARMUP_RATIO,
    TRANSFORMER_WEIGHT_DECAY,
)
from src.transformer import TextClassificationDataset, WeightedTrainer, compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--model-name", default="ufal/robeczech-base")
    parser.add_argument("--max-length", type=int, default=TRANSFORMER_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=TRANSFORMER_LR)
    parser.add_argument("--epochs", type=int, default=TRANSFORMER_EPOCHS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on rows (for smoke tests).")
    return parser.parse_args()

def main():
    args = parse_args()


    