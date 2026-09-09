"""fastText embeddings as a fixed representation for the classic classifiers."""
import numpy as np
from sklearn.metrics import f1_score

from src.classifiers import CLASSIFIER_CONFIGS

# A comprehension is used instead of deleting keys so the shared CLASSIFIER_CONFIGS stays intact
EMBEDDING_CONFIGS = {k: v for k, v in CLASSIFIER_CONFIGS.items()
                     if k not in ("multinomial_nb", "complement_nb")}

def _import_fasttext():
    """Import fasttext lazily so the TF-IDF and transformer pipelines run without it."""
    try:
        import fasttext
    except ImportError as exc:
        raise ImportError(
            "fastText is optional. Install with: pip install -e .[fasttext]"
        ) from exc
    return fasttext


def load_fasttext(model_path):
    """Load a pretrained fastText model"""
    fasttext = _import_fasttext()
    return fasttext.load_model((str(model_path))) # fasttext.load_model can accept only strings - not Path objects

def embed_documents(texts, model):
    """Take a pandas series and return a list of vectors calculated using fastText model

    Args:
        texts: Iterable of document strings - a pandas Series.
        model: Loaded fastText model.

    Returns:
        Array of shape (n_documents, 300). Newlines are replaced with spaces
        because get_sentence_vector treats them as sentence separators and
        raises on multi-line input.
    """
    vectors = []
    for text in texts:
        clean_text = text.replace("\n", " ") #fastText fails on newline characters
        vectors.append(model.get_sentence_vector(clean_text))
    return np.vstack(vectors)


def objective(trial, config, X_train, y_train, X_val, y_val):
    """Sample classifier hyperparameters, fit on fixed embeddings, scored on validation set.

    Args:
        trial: Optuna trial used to sample the search space.
        config: ClassifierConfig for the classifier being tuned.
        X_train: Training embeddings of shape (n_train, 300).
        y_train: Integer training labels.
        X_val: Validation embeddings of shape (n_val, 300).
        y_val: Integer validation labels.
        class_weight: Optional {label: weight} dict forwarded to
            `config.build`. See `ClassifierConfig.build`.

    Returns:
        Macro-F1 on the validation data.
    """
    classifier = config.build(config.sample(trial), class_weight=class_weight)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_val)
    return f1_score(y_val, predictions, average="macro")
    return f1_score(y_val, predictions, average="macro")
