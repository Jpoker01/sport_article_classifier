"""fastText embeddings for representation + classic classifier set-up from the set-up in traditional.py"""
import numpy as np
import fasttext
from sklearn.metrics import f1_score

from src.classifiers import CLASSIFIER_CONFIGS

EMBEDDING_CONFIGS = {k: v for k, v in CLASSIFIER_CONFIGS.items()
                     if k not in ("multinomial_nb", "complement_nb")}