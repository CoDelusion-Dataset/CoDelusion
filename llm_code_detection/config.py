"""
Configuration parameters for the CodeBERT classifier.
"""

import logging
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_MODEL_PATH = './codebert-base'
DEFAULT_MAX_LEN = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WARMUP_STEPS = 0
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 4
DEFAULT_OUTPUT_DIR = './output'

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 