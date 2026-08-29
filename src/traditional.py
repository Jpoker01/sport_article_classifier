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


CLASSIFIER_CONFIGS = {
    "logreg": ClassifierConfig(
        classifier_object=LogisticRegression,
        searched_hyperparameters={
            "C": lambda t: t.suggest_float("C", 1e-2, 1e2, log=True),
        },
        fixed_hyperparameters={
            "class_weight": "balanced", "max_iter": 1000, "random_state": SEED,
        },
    ),
    "linear_svc": ClassifierConfig(
        classifier_object=LinearSVC,
        searched_hyperparameters={
            "C": lambda t: t.suggest_float("C", 1e-2, 1e2, log=True),
            "loss": lambda t: t.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        },
        fixed_hyperparameters={
            "class_weight": "balanced", "max_iter": 5000, "random_state": SEED,
        },
    ),
    "multinomial_nb": ClassifierConfig(
        classifier_object=MultinomialNB,
        searched_hyperparameters={
            "alpha": lambda t: t.suggest_float("alpha", 1e-3, 10.0, log=True),
            "fit_prior": lambda t: t.suggest_categorical("fit_prior", [True, False]),
        },
    ),
    "complement_nb": ClassifierConfig(
        classifier_object=ComplementNB,
        searched_hyperparameters={
            "alpha": lambda t: t.suggest_float("alpha", 1e-3, 10.0, log=True),
            "fit_prior": lambda t: t.suggest_categorical("fit_prior", [True, False]),
            "norm": lambda t: t.suggest_categorical("norm", [True, False]),
        },
    ),
    "random_forest": ClassifierConfig(
        classifier_object=RandomForestClassifier,
        searched_hyperparameters={
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 400),
            "max_depth": lambda t: t.suggest_int("max_depth", 10, 60),
            "max_features": lambda t: t.suggest_categorical("max_features", ["sqrt", "log2"]),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 10),
        },
        fixed_hyperparameters={
            "class_weight": "balanced", "n_jobs": -1, "random_state": SEED,
        },
    ),
    "xgboost": ClassifierConfig(
        classifier_object=XGBClassifier,
        searched_hyperparameters={
            "n_estimators": lambda t: t.suggest_int("n_estimators", 200, 600),
            "max_depth": lambda t: t.suggest_int("max_depth", 4, 12),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
            "subsample": lambda t: t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
        },
        fixed_hyperparameters={
            "tree_method": "hist", "n_jobs": -1, "random_state": SEED,
            # "device": "cuda",  # uncomment to run on the A40 GPU
        },
    ),
}

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