"""fastText embeddings as a fixed representation for the classic classifiers."""
import numpy as np
from sklearn.metrics import f1_score
from gensim.models.fasttext import load_facebook_vectors

from src.classifiers import CLASSIFIER_CONFIGS

# A comprehension is used instead of deleting keys so the shared CLASSIFIER_CONFIGS stays intact
EMBEDDING_CONFIGS = {k: v for k, v in CLASSIFIER_CONFIGS.items()
                     if k not in ("multinomial_nb", "complement_nb")}

def load_fasttext(model_path):
    return load_facebook_vectors(str(model_path))

def embed_documents(texts, kv):
    dim = kv.vector_size
    vectors = []
    for text in texts:
        tokens = text.replace("\n", " ").split()
        # fastText-style: L2-normalize each word vector, then average
        vecs = []
        for t in tokens:
            v = kv[t]  # gensim handles OOV via subwords automatically
            n = np.linalg.norm(v)
            if n > 0:
                vecs.append(v / n)
        vec = np.mean(vecs, axis=0) if vecs else np.zeros(dim)
        vectors.append(vec)
    return np.vstack(vectors)

def objective(trial, config, X_train, y_train, X_val, y_val, class_weight=None):
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
