from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_DATA_PATH = DATA / "sportoclanky.csv.gz"
CLEAN_DATA_PATH = DATA / "clean.parquet"

MIN_CLASS_COUNT = 10 

# split
VAL_SIZE = 0.1
TEST_SIZE = 0.1
