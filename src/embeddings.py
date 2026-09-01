"""fastText embeddings for representation + classic classifier set-up from the set-up in traditional.py"""
import numpy as np
import fasttext
from sklearn.metrics import f1_score

from src.classifiers import CLASSIFIER_CONFIGS

# comprehension just in case we ran both run_traditional and run_embeddings files
EMBEDDING_CONFIGS = {k: v for k, v in CLASSIFIER_CONFIGS.items()
                     if k not in ("multinomial_nb", "complement_nb")}


def load_fasttext(model_path):
    """Load a pretrained fasText model"""
    return fasttext.load_model((str(model_path))) # fasttext.load_model can accept only strings - not Path objects

def embed_documents(texts, model):
    """Average subword word vectors into one 300-dim vector per document."""
    vectors = []
    for text in texts:
        clean_text = text.replace("\n", " ")
        vectors.append(model.get_sentence_vector(clean_text))
    return np.vstack(vectors)


def objective(trial, config, X_train, y_train, X_val, y_val):
    """Sample classifier hyperparameters, fit on fixed embeddings, score macro-F1 on val."""
    classifier = config.build(trial)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_val)
    return f1_score(y_val, predictions, average="macro")