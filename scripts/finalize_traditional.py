"""Retrain the traditional combination winner (TF-IDF + classifier) on train and evaluate on test dataset."""
import sys
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifiers import CLASSIFIER_CONFIGS
from src.config import CLEAN_DATA_PATH, ROOT
from src.evaluate import evaluate, full_report
from src.traditional import build_tfidf_vectorizer