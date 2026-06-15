import os

DATA_DIR = "data"
BATCH_SIZE = 128
NUM_WORKERS = 0 if os.path.exists("/content") else 2
