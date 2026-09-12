# Czech Sport Article Classifier

Machine learning project focused on training an optimal classifier model for Czech sport news
articles. Three approaches are compared: 
* TF-IDF in combination with traditional classifiers.
* Pretrained fastText embeddings with the same traditional classifiers.
* Fine-tuned czech transformer.
  
All hyperparameters are tuned with Optuna against validation macro-F1.  

## Results

| Method | Macro-F1 | Accuracy | Inference | Size |
|---|---|---|---|---|
| **RobeCzech, fine-tuned** | **97.3%** | **99.2%** | 1.5 ms/sample (A40) | 484 MB |
| TF-IDF + LinearSVC | 96.7% | 98.9% | 0.22 ms/sample (CPU) | 16 MB |
| fastText + LinearSVC | 90.5% | 96.9% | 0.24 ms/sample (CPU) | ~7 GB |

Macro-F1 is the primary metric because the classes are heavily imbalanced.

## Tech Stack
*   **Language:** Python 3.11 to 3.12 (results produced on 3.12)
*   **Methods**
    *   **TF-IDF (scikit-learn):** Optuna selects between word-level unigrams/bigrams and character n-grams (`analyzer="char_wb"`), sparse features.
    *   **fastText:** Pretrained Czech 300-dimensional embeddings (`cc.cs.300.bin`), averaged into one document vector.
    *   **Transformers:** Two Czech encoders tried, both fine-tuned end to end (all parameters trainable):
        *   **RobeCzech (`ufal/robeczech-base`):** Czech RoBERTa. Selected for final results.
*   **Classifiers:** scikit-learn classifiers such as LogisticRegression, LinearSVC, MultinomialNB, ComplementNB, RandomForest and XGBoost.
*   **Hyperparameter tuning:** Optuna, per-classifier studies, macro-F1 as the objective.
*   **Environment:** Jupyter for analysis and results, PyCharm for development.

## Quickstart guide

### Software requirements
- Python 3.11 to 3.12
- Git
- Linux, macOS or Windows. The TF-IDF and transformer pipelines need nothing beyond the above.

### Hardware requirements
All code was executed inside of the MetaCentrum Jupyter environment:
- **GPU:** NVIDIA A40 (48 GB VRAM)
- **CPU:** 16 cores
- **RAM:** 64 GB
- **Disk:** ~30 GB free

Minimum:
- **CPU-only (TF-IDF and fastText paths):** 8 GB RAM, 15 GB free disk. The fastText binary alone occupies ~7 GB in memory once loaded.
- **GPU (transformer path):** NVIDIA GPU with at least 16 GB VRAM for training at batch size 16 in mixed precision. Inference alone runs comfortably on 8 GB VRAM. Training on CPU is possible but not practical (multiple hours per epoch).

### Setup

The project installs as an editable package, which registers `src/` on the import
path (so `from src.config import ...` works from anywhere) and pulls in all
dependencies from `pyproject.toml`.

**Setup commands**

```bash
git clone https://github.com/Jpoker01/sport_article_classifier.git
cd sport_article_classifier
pip install -e .
```

### Data preparation

**The raw dataset is not included in this repository, as the source cannot be disclosed.**  
Place the raw dataset into `data/` (kept out of git). The `data_prep.ipynb` notebook cleans it and writes `data/clean.parquet`, which every training script reads and uses.

The Czech fastText model needs to be downloaded separately (~7 GB):

```bash
mkdir -p data
wget -O data/cc.cs.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.bin.gz
gunzip data/cc.cs.300.bin.gz
```
Notebook order:
1. `notebooks/data_analysis.ipynb` - exploratory analysis of the raw dataset.
2. `notebooks/data_prep.ipynb` - cleaning, deduplication, stratified 80/10/10 split.

### Running the experiments

Each Optuna study runs `N_TRIALS` trials per classifier (or per configuration for the transformer). Higher values explore the hyperparameter space more thoroughly at proportional cost. Reported numbers use 50 trials per classifier for TF-IDF and fastText, and 38 trials for the transformer (limited by time and memory constraints).

**Substitute `N_TRIALS` with the value you want to use.**

**TF-IDF + classical classifiers**
```bash
python -m scripts.run_traditional --n-trials N_TRIALS
python -m scripts.finalize_traditional
```

**fastText embeddings + classical classifiers**
```bash
python -m scripts.run_embeddings --n-trials N_TRIALS
python -m scripts.finalize_embeddings
```

**Fine-tuned transformer**

The reported results use `ufal/robeczech-base`. The `--model-name` argument accepts any Hugging Face model identifier compatible with `AutoModelForSequenceClassification`.

```bash
python -m scripts.tune_transformer --model-name ufal/robeczech-base --n-trials N_TRIALS
python -m scripts.run_transformer --model-name ufal/robeczech-base
python -m scripts.finalize_transformer --model-dir results/transformer/ufal_robeczech-base
```

### Results analysis

Once all three finalize scripts have run, open `notebooks/results_analysis.ipynb` for the final comparison table, confusion matrices, per-class F1 and error analysis.

## Project structure

The work is organized as follows:

 * **pyproject.toml** - Dependencies and package configuration for `pip install -e .`
 * **/data** - Raw and processed datasets - **not included in git**
 * **/notebooks** - Jupyter notebooks
   * **data_analysis.ipynb** - Exploratory analysis of the raw dataset
   * **data_prep.ipynb** - Cleaning, deduplication and stratified split
   * **tokenization_analysis.ipynb** - token-length check for the transformer encoders
   * **results_analysis.ipynb** - final comparison, figures and error analysis
 * **/src** - Importable package, no scripts
   * **config.py** - Paths, seed, split sizes, shared constants
   * **classifiers.py** - Classifier configurations and Optuna search spaces
   * **traditional.py** - TF-IDF representation and its Optuna objective
   * **embeddings.py** - fastText representation and its Optuna objective
   * **transformer.py** - Dataset, metrics, weighted trainer, Optuna objective
   * **weighting.py** - capped inverse-frequency class weights
   * **evaluate.py** - Shared metric and reporting helpers
* **scripts/** - entry points (`run_*`/`tune_*` search, `finalize_*` refit and evaluate on test)
* **results/** - metrics, predictions and figures per method. Trained weights are not in git; the transformer weights are on the Hugging Face Hub.

