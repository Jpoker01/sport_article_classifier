"""Traditional TF-IDF baselines with Optuna tuning."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

def build_tfidf_vectorizer(max_features, min_df, ngram_range=(1, 2), 
                           lowercase=True, sublinear_tf=True):
    """Build a word-level TF-IDF vectorizer.

    Args:
        max_features: Hard limit on vocabulary size of the whole corpus.
        min_df: Minimum number of documents a term must appear in to be kept in the vocabulary.
        ngram_range: Lower and upper bound on n-gram size.
        lowercase: Whether to lowercase text before tokenizing.
        sublinear_tf: If True, replace raw term frequency with 1 + log(tf) 
        which gradually lowers the weight of very frequent terms.

    Returns:
        Unfitted TfidfVectorizer set to specified parameters.
    """
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        lowercase=lowercase,
        sublinear_tf=sublinear_tf
    )

def objective(trial, config, train_texts, y_train, val_texts, y_val):
    """Sample vectorizer and classifier hyperparameters, fit, and use validation set to score.

    Args:
        trial: Optuna trial used to sample the search space.
        config: ClassifierConfig for the classifier being tuned.
        train_texts: Training documents.
        y_train: Integer training labels.
        val_texts: Validation documents.
        y_val: Integer validation labels.

    Returns:
        Macro-F1 on the validation data.
    """

    # Vectorizer params carry a tfidf_ prefix to avoid an Optuna name
    # collision with classifier params (RandomForest also has max_features).
    vectorizer = build_tfidf_vectorizer(
    ngram_range=(1, trial.suggest_categorical("tfidf_ngram_max", [1, 2])),
    max_features=trial.suggest_categorical("tfidf_max_features", [20000, 50000, 100000]),
    min_df=trial.suggest_int("tfidf_min_df", 1, 5),
    sublinear_tf=trial.suggest_categorical("tfidf_sublinear_tf", [True, False]),
    )

    
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    classifier = config.build(config.sample(trial))
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_val)
    
    return f1_score(y_val, predictions, average="macro")