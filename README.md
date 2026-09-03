# Czech Sport Article Classifier

This project classifies Czech sport news articles into multiple sport categories based on the article title and perex. Three approaches are compared: 
* TF-IDF with classical classifiers
* Pretrained fastText embeddings with the same classifier family
* Fine-tuned Czech transformer.
  
All hyperparameters are tuned with Optuna against validation macro-F1.  

## Results

| Method | Macro-F1   | Accuracy   | Inference (ms/sample) | Model size |
|--------|------------|------------|-----------------------|------------|
| TF-IDF + LinearSVC | 0.9183     | 0.9846     | 0.04 | 11 MB |
| fastText + XGBoost | 0.8024     | 0.9644     | 0.16 | ~7 GB |
| RobeCzech, fine-tuned | **0.9711** | **0.9913** | 3.41 | 484 MB |

Macro-F1 is the primary metric because the classes are heavily imbalanced.

## Tech Stack
*   **Language:** Python 3.13
*   **Text representations:**
    *   **TF-IDF (scikit-learn):** Optuna selects between word-level unigrams/bigrams and character n-grams (`analyzer="char_wb"`), sparse features.
    *   **fastText:** Pretrained Czech 300-dimensional embeddings (`cc.cs.300.bin`), averaged into one document vector.
    *   **Transformers:** Two Czech encoders tried, both fine-tuned end to end (all parameters trainable):
        *   **RobeCzech (`ufal/robeczech-base`):** Czech RoBERTa. Selected for final results.
        *   **Small-E-Czech (`Seznam/small-e-czech`):** Czech ELECTRA. Tried and dropped after reaching only 0.63 validation macro-F1 with the same recipe.  
*   **Classifiers:** scikit-learn classifiers such as LogisticRegression, LinearSVC, MultinomialNB, ComplementNB, RandomForest and then XGBoost.
*   **Hyperparameter tuning:** Optuna, per-classifier studies, macro-F1 as the objective.
*   **Environment:** Jupyter for analysis and results, PyCharm for development.

## Quickstart guide

### Software requirements
- Python 3.13
- Git
- Linux, macOS or WSL2 on Windows

### Hardware requirements
All code was executed on the MetaCentrum Jupyter environment:
- **GPU:** NVIDIA A40 (48 GB VRAM)
- **CPU:** 16 cores
- **RAM:** 64 GB
- **Disk:** ~30 GB free

Minimum:
- **CPU-only (TF-IDF and fastText paths):** 8 GB RAM, 10 GB free disk. The fastText binary alone occupies ~7 GB in memory once loaded.
- **GPU (transformer path):** NVIDIA GPU with at least 16 GB VRAM for training at batch size 16 in mixed precision. Inference alone runs comfortably on 8 GB VRAM. Training on CPU is possible but not practical (multiple hours per epoch).

### Setup

```bash
git clone https://github.com/Jpoker01/sport_article_classifier.git
cd sport_article_classifier
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Data preparation

The raw dataset is not included in this repository, as the source cannot be disclosed. Place the raw dataset into `data/` (kept out of git). The `data_prep.ipynb` notebook cleans it and writes `data/clean.parquet`, which every training script reads and uses.

The Czech fastText model needs to be downloaded separately (~7 GB):

```bash
mkdir -p data
wget -O data/cc.cs.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.bin.gz
gunzip data/cc.cs.300.bin.gz
```
Notebook order:
1. `notebooks/data_analysis.ipynb` - exploratory analysis of the raw dataset.
2. `notebooks/data_prep.ipynb` - cleaning, deduplication, stratified 70/15/15 split, fastText download.

### Running the experiments

Each Optuna study runs `N_TRIALS` trials per classifier (or per configuration for the transformer). Higher values explore the hyperparameter space more thoroughly at proportional cost. Reported numbers use **50** trials per classifier for all methods.

**Substitute `N_TRIALS` with the value you want to use.**

**TF-IDF + classical classifiers**
```bash
python scripts/run_traditional.py --n-trials N_TRIALS
python scripts/finalize_traditional.py
```

**fastText embeddings + classical classifiers**
```bash
python scripts/run_embeddings.py --n-trials N_TRIALS
python scripts/finalize_embeddings.py
```

**Fine-tuned transformer**

The reported results use `ufal/robeczech-base`. `Seznam/small-e-czech` was also tried but dropped at 0.63 validation macro-F1. The `--model-name` argument accepts any Hugging Face model identifier compatible with `AutoModelForSequenceClassification`.

```bash
python scripts/tune_transformer.py --model-name ufal/robeczech-base --n-trials N_TRIALS
python scripts/run_transformer.py --model-name ufal/robeczech-base --run-name tuned \
    --lr <LR> --batch-size <BATCH_SIZE> --weight-decay <WEIGHT_DECAY> --label-smoothing <LABEL_SMOOTHING>
python scripts/finalize_transformer.py --model-dir results/transformer/ufal_robeczech-base__tuned/best
```

Replace `<LR>`, `<BATCH_SIZE>`, `<WEIGHT_DECAY>` and `<LABEL_SMOOTHING>` with the winning row from `results/transformer/ufal_robeczech-base_optuna_trials.csv`.
### Results analysis

Once all three finalize scripts have run, open `notebooks/results_analysis.ipynb` for the final comparison table, confusion matrices, per-class F1 and error analysis.

## Project structure

The work is organized as follows:

  * **pyproject.toml** - Dependencies and package configuration for `pip install -e .`
 * **/data** - Raw and processed datasets - **not included in git**
 * **/notebooks** - Jupyter notebooks
   * **data_analysis.ipynb** - Exploratory analysis of the raw dataset
   * **data_prep.ipynb** - Cleaning, deduplication and stratified split
   * **results_analysis.ipynb** - Final comparison, figures and error analysis
 * **/src** - Importable package, no scripts
   * **config.py** - Paths, seed, split sizes, shared constants
   * **classifiers.py** - Classifier configurations and Optuna search spaces
   * **traditional.py** - TF-IDF representation and its Optuna objective
   * **embeddings.py** - fastText representation and its Optuna objective
   * **transformer.py** - Dataset, metrics, weighted trainer, Optuna objective
   * **evaluate.py** - Shared metric and reporting helpers
 * **/scripts** - Entry points for experiments
   * **run_traditional.py** - Optuna search over TF-IDF classifiers
   * **run_embeddings.py** - Optuna search over fastText classifiers
   * **tune_transformer.py** - Optuna search over transformer fine-tuning hyperparameters
   * **run_transformer.py** - Single transformer fine-tuning run (based on the winner combination)
   * **finalize_traditional.py** - Refits the winning TF-IDF configuration on train+val and evaluates it on the test split
   * **finalize_embeddings.py** - Refits the winning fastText classifier on train+val and evaluates it on the test split
   * **finalize_transformer.py** - Loads the fine-tuned transformer from `--model-dir` and evaluates it on the test split
 * **/results** - Metrics, predictions and figures. Trained model weights are **not included in git** due to size and to avoid disclosing any data information.
