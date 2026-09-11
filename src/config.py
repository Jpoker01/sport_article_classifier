from pathlib import Path
import torch

#check for GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Seed for reproducility of experiments
SEED = 42

# Data paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"

RAW_DATA_PATH = DATA_PATH / "sportoclanky.csv.gz"
CLEAN_DATA_PATH = DATA_PATH / "clean.parquet"
FASTTEXT_MODEL_PATH = DATA_PATH / "cc.cs.300.bin"

RESULTS_PATH = ROOT / "results"
+TRADITIONAL_DIR = RESULTS_PATH / "traditional"
+EMBEDDINGS_DIR = RESULTS_PATH / "embeddings"

# Training / test / validation split constants
VAL_SIZE = 0.1
TEST_SIZE = 0.1

#Weighting
CLASS_WEIGHT_BETA = 0.5
CLASS_WEIGHT_CAP = 8.0

# Evaluation
PRIMARY_METRIC = "macro_f1"

# Transformer defaults 
TRANSFORMER_MAX_LENGTH = 256
TRANSFORMER_LR = 2e-5
TRANSFORMER_EPOCHS = 4
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_RATIO = 0.1
TRANSFORMER_EARLY_STOPPING_PATIENCE = 2