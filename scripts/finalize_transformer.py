"""Evaluate a fine-tuned transformer on the test data set."""


import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CLEAN_DATA_PATH, DEVICE, TRANSFORMER_MAX_LENGTH
from src.evaluate import evaluate, full_report
from src.transformer import TextClassificationDataset

def parse_args():
    """Parse command line arguments.

    Returns:
        Namespace with model_dir, max_length and batch_size.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-length", type=int, default=TRANSFORMER_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()

def main():
    """Score a saved transformer on the held-out test split and save the results."""
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = model_dir.parent

    df = pd.read_parquet(CLEAN_DATA_PATH)
    test = df[df["split"] == "test"]

    encoder = LabelEncoder()
    encoder.classes_ = np.load(out_dir / "label_classes.npy", allow_pickle=True)
    y_test = encoder.transform(test["category"])

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(DEVICE).eval()

    dataset = TextClassificationDataset(test["text"], y_test, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    elapsed = time.time() - start

    y_pred = np.array(predictions)

    metrics = evaluate(y_test, y_pred)
    metrics["inference_seconds"] = elapsed
    metrics["inference_ms_per_sample"] = elapsed / len(test) * 1000
    
    print(metrics)
    labels = np.arange(len(encoder.classes_))
    confusion = full_report(y_test, y_pred, labels=labels,
                            target_names=encoder.classes_)

    metrics_path = out_dir / "test_metrics.csv"
    predictions_path = out_dir / "test_predictions.csv"
    confusion_path = out_dir / "test_confusion_matrix.csv"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame({
        "text": test["text"].values,
        "true": encoder.inverse_transform(y_test),
        "pred": encoder.inverse_transform(y_pred),
    }).to_csv(predictions_path, index=False)
    pd.DataFrame(confusion, index=encoder.classes_,
                 columns=encoder.classes_).to_csv(confusion_path)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {confusion_path}")


if __name__ == "__main__":
    main()