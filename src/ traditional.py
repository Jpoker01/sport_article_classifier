"""Traditional TF-IDF baselines with Optuna tuning."""
from dataclasses import dataclass, field

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB #complementNB added as it is preferred for imbalanced datasets
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score


from src.config import SEED
