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

@dataclass
class ClassifierConfig:
    """Wraps a classifier class together with its Optuna search space."""
    classifier_object: type
    searched_hyperparameters: dict = field(default_factory=dict)
    fixed_hyperparameters: dict = field(default_factory=dict)

    def build(self, trial):
        """Instantiate the classifier with sampled + fixed hyperparameters."""
        sampled = {
            name: suggest(trial)
            for name, suggest in self.searched_hyperparameters.items()
        }
        return self.classifier_object(**sampled, **self.fixed_hyperparameters)


def build_tfidf_vectorizer(max_features, min_df, lowercase=True, sublinear_tf=True):
    """Word-level TF-IDF vectorizer (unigrams + bigrams)."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        lowercase=lowercase,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )

def objective(trial, config, train_texts, y_train, val_texts, y_val):
    """Sample vectorizer + classifier hyperparameters, fit, score macro-F1 on val."""
    vectorizer = build_tfidf_vectorizer(
        max_features=trial.suggest_categorical("max_features", [20000, 50000, 100000]),
        min_df=trial.suggest_int("min_df", 1, 5),
        sublinear_tf=trial.suggest_categorical("sublinear_tf", [True, False]),
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    classifier = config.build(trial)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_val)
    return f1_score(y_val, predictions, average="macro")