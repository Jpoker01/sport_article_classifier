"""Traditional TF-IDF baselines with Optuna tuning."""
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

def build_tfidf_vectorizer(max_features, min_df, ngram_range=(1, 2),
                           lowercase=True, sublinear_tf=True, analyzer="word"):
    """Build a TF-IDF vectorizer over words or character n-grams.
 
    Args:
        max_features: Hard limit on vocabulary size of the whole corpus.
        min_df: Minimum number of documents a term must appear in to be kept in the vocabulary.
        ngram_range: Lower and upper bound on n-gram size.
        lowercase: Whether to lowercase text before tokenizing.
        sublinear_tf: If True, replace raw term frequency with 1 + log(tf)
        which gradually lowers the weight of very frequent terms.
        analyzer: "word" for word n-grams, "char_wb" for character n-grams
        taken inside word boundaries. Character n-grams share substrings
        between inflected forms of the same word, which is what a stemmer
        would otherwise provide.
 
    Returns:
        Unfitted TfidfVectorizer set to specified parameters.
    """
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        lowercase=lowercase,
        sublinear_tf=sublinear_tf
    )
 
def objective(trial, config, train_texts, y_train, val_texts, y_val, class_weight=None):
    """Sample vectorizer and classifier hyperparameters, fit, and use validation set to score.
 
    Args:
        trial: Optuna trial used to sample the search space.
        config: ClassifierConfig for the classifier being tuned.
        train_texts: Training documents.
        y_train: Integer training labels.
        val_texts: Validation documents.
        y_val: Integer validation labels.
        class_weight: Optional {label: weight} dict forwarded to
            `config.build`. See `ClassifierConfig.build`.
 
    Returns:
        Macro-F1 on the validation data.
    """

    #character n-grams sampled only for classifiers whose config allows them 
    analyzer = "word"
    if config.allows_char_ngrams:
        analyzer = trial.suggest_categorical("tfidf_analyzer", ["word", "char_wb"])
 
    if analyzer == "word":
        ngram_range = (1, trial.suggest_categorical("tfidf_ngram_max", [1, 2]))
    else:
        ngram_range = (3, trial.suggest_categorical("tfidf_char_ngram_max", [4, 5]))
 
    # Vectorizer params carry a tfidf_ prefix to avoid an Optuna name
    # collision with classifier params (RandomForest also has max_features).
    vectorizer = build_tfidf_vectorizer(
    analyzer=analyzer,
    ngram_range=ngram_range,
    max_features=trial.suggest_categorical("tfidf_max_features", [20000, 50000, 100000]),
    min_df=trial.suggest_int("tfidf_min_df", 1, 5),
    sublinear_tf=trial.suggest_categorical("tfidf_sublinear_tf", [True, False]),
    )
 
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
 
    classifier = config.build(config.sample(trial), class_weight=class_weight)
    fit_kwargs = {}
    if isinstance(classifier, XGBClassifier) and class_weight is not None:
        fit_kwargs["sample_weight"] = np.array(
            [class_weight[int(y)] for y in y_train]
        )

    classifier.fit(X_train, y_train, **fit_kwargs)
    predictions = classifier.predict(X_val)
    return f1_score(y_val, predictions, average="macro")
    
