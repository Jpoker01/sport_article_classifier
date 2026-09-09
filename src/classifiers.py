"""Shared file for 'traditional' classifiers and their Optuna search set-ups"""
from dataclasses import dataclass, field
from ast import literal_eval
import pandas as pd
 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB #complementNB added as it is preferred for imbalanced datasets
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
 
from src.config import SEED, DEVICE
 
@dataclass
class ClassifierConfig:
    """Wraps a classifier class together with its Optuna search space.
 
    Attributes:
        classifier_object: classifier class
        searched_hyperparameters: Maps a parameter name to a callable that samples it from an Optuna trial.
        fixed_hyperparameters: Parameters passed unchanged on every build.
        allows_char_ngrams: Whether this classifier may be offered character n-grams. Tree ensembles are excluded, the denser matrix makes them too slow.
    """
    classifier_object: type
    searched_hyperparameters: dict = field(default_factory=dict)
    fixed_hyperparameters: dict = field(default_factory=dict)
    allows_char_ngrams: bool = False
 
    def sample(self, trial):
        """Draw one set of hyperparameters from the search space.
 
        Args:
            trial: Optuna trial used to sample each searched parameter.
 
        Returns:
            Dict mapping parameter names to the sampled values.
        """
        return {
            name: suggest(trial)
            for name, suggest in self.searched_hyperparameters.items()
        }
 
    def build(self, params):
        """Instantiate the classifier from given params plus the fixed ones.
 
        Args:
            params: Hyperparameters to pass to the classifier, either from
            `sample()` during tuning or parsed from a trials CSV during
            finalization.
             class_weight: Optional {label: weight} dict replacing the classifier's default `class_weight` 
             where supported (LogisticRegression, LinearSVC, RandomForest). 
             Silently ignored for classifiers without a `class_weight` parameter.
        Returns:
            An unfitted classifier instance.
        """
        fixed = dict(self.fixed_hyperparameters)
        if class_weight is not None and "class_weight" in fixed:
            fixed["class_weight"] = class_weight
        return self.classifier_object(**params, **fixed)
 
CLASSIFIER_CONFIGS = {
    "logreg": ClassifierConfig(
        classifier_object=LogisticRegression,
        searched_hyperparameters={
            "C": lambda t: t.suggest_float("C", 1e-2, 1e2, log=True),
        },
        fixed_hyperparameters={
            "class_weight": "balanced", "max_iter": 1000, "random_state": SEED,
        },
        allows_char_ngrams=True,
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
        allows_char_ngrams=True,
    ),
    "multinomial_nb": ClassifierConfig(
        classifier_object=MultinomialNB,
        searched_hyperparameters={
            "alpha": lambda t: t.suggest_float("alpha", 1e-3, 10.0, log=True),
            "fit_prior": lambda t: t.suggest_categorical("fit_prior", [True, False]),
        },
        allows_char_ngrams=True,
    ),
    "complement_nb": ClassifierConfig(
        classifier_object=ComplementNB,
        searched_hyperparameters={
            "alpha": lambda t: t.suggest_float("alpha", 1e-3, 10.0, log=True),
            "fit_prior": lambda t: t.suggest_categorical("fit_prior", [True, False]),
            "norm": lambda t: t.suggest_categorical("norm", [True, False]),
        },
        allows_char_ngrams=True,
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
            "device": DEVICE
        },
    ),
}
 
def parse_value(value):
    """Turn a CSV cell back into its Python type; leave plain strings as-is.
 
    Args:
        value: Cell value read from a trials CSV.
 
    Returns:
        The value as bool, int, float or str. Whole-number floats are
        downcast to int because pandas widens integer columns to float64
        when other rows hold NaN, and classifiers such as XGBoost reject
        float values for integer hyperparameters like n_estimators.
    """
    if pd.isna(value):
        return None
    try:
        parsed = literal_eval(str(value))
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed, float) and parsed.is_integer():
        return int(parsed)
    return parsed
 
def load_best_trial(trials_path, metric_column="best_val_macro_f1") -> tuple:
    """Return the best-scoring trial recorded in an Optuna trials CSV.
 
    Args:
        trials_path: Path to a CSV whose index is the classifier name and
        whose columns hold the metric plus sampled hyperparameters.
        metric_column: Column to rank trials by, higher is better.
 
    Returns:
        Tuple of (classifier name, metric value, hyperparameter dict).
        NaN columns are dropped from the dict because different
        classifiers have different search spaces, so the CSV is sparse.
    """
    trials = pd.read_csv(trials_path, index_col=0)
    row = trials.sort_values(metric_column, ascending=False).iloc[0]
    params = {
        column: parse_value(value)
        for column, value in row.items()
        if column != metric_column and not pd.isna(value)
    }
    return row.name, row[metric_column], params
 

