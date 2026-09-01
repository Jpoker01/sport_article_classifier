from pathlib import Path

#Seed for reproducility
SEED = 42

# Data paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_DATA_PATH = DATA / "sportoclanky.csv.gz"
CLEAN_DATA_PATH = DATA / "clean.parquet"
FASTTEXT_MODEL_PATH = DATA / "cc.cs.300.bin"

# Training / test / validation split constants.
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# Evaluation
PRIMARY_METRIC = "macro_f1"

# Transformer defaults 
TRANSFORMER_MAX_LENGTH = 256
TRANSFORMER_LR = 2e-5
TRANSFORMER_EPOCHS = 4
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_RATIO = 0.1
TRANSFORMER_EARLY_STOPPING_PATIENCE = 2