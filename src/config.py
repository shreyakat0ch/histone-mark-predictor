import os

# --- Paths ---
# By default, we look for data in the project root.
# You can override these using environment variables.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root directory for raw data (BED and RNA files)
DATA_DIR = os.getenv("HISTONE_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

# Directory for processed caches and metadata
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")

# Output directories
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
PLOTS_DIR = os.path.join(RUNS_DIR, "plots")

# Genomic References
FASTA_PATH = os.getenv("HISTONE_FASTA_PATH", os.path.join(DATA_DIR, "hg38.fa"))
GTF_PATH = os.getenv("HISTONE_GTF_PATH", os.path.join(DATA_DIR, "hg38.knownGene.gtf"))

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Data Settings ---
# Marks selected for TF binding prediction:
#   ENHANCING marks  → chromatin open, TFs freely bind
#   INHIBITING marks → chromatin compacted, TFs blocked
TARGET_MARKS = [
    "H3K27ac",   # ENHANCES TF binding — active enhancers + promoters
    "H3K4me3",   # ENHANCES TF binding — active promoters (sharp, sequence-driven)
    "H3K27me3",  # INHIBITS TF binding — Polycomb repression
    "H3K9me3",   # INHIBITS TF binding — constitutive heterochromatin
    "H3K4me1",
]
NUM_MARKS = len(TARGET_MARKS)  # = 5

# 8192bp windows capture broad enhancer/repressor context for H3K27ac and H3K27me3
WINDOW_SIZE = 32768
HALF_WINDOW = WINDOW_SIZE // 2

# Top most variable genes to use as cell-line RNA fingerprint
RNA_NUM_TOP_GENES = 4000
# Since we are peak-centered, there is no single "local gene". 
# The model just takes the global fingerprint.
RNA_INPUT_DIM = RNA_NUM_TOP_GENES

# --- Model Architecture Settings ---
# DNA Encoder (CNN + Transformer)
USE_DILATED_CONVS = False
CNN_DILATIONS = [2, 4, 8]
CNN_CHANNELS = [4, 128, 256, 256] # Reduced last layer to match D_MODEL
CNN_KERNELS = [15, 9, 9]
CNN_STRIDES = [1, 4, 4]

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4  # Reduced from 6 for faster training
D_FEEDFORWARD = 1024
DROPOUT = 0.1

# RNA Encoder (MLP)
RNA_HIDDEN_DIM = 512

# Set True for an RNA ablation experiment. This zeros the RNA vector during
# train/validation/evaluation so performance can be compared against the
# full RNA-conditioned model.
RNA_ABLATION = False

# Optional mark-specific context pooling experiment. When enabled, the shared
# encoder still sees the full WINDOW_SIZE, but each mark head pools over a
# different central span of encoded tokens.
USE_MARK_SPECIFIC_CONTEXT = True
MARK_CONTEXT_BP = {
    "H3K27ac": 8192,
    "H3K4me3": 4096,
    "H3K27me3": 32768,
    "H3K9me3": 16384,
    "H3K4me1": 16384,
}

# Multitask Head
HEAD_HIDDEN_DIM = 512

# --- Training Settings ---
BATCH_SIZE = 128 # Higher throughput on large peak-sampled batches
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 50
WARMUP_STEPS = 1000
EARLY_STOPPING_PATIENCE = 15

# V7 Data Loading Settings
PEAK_FRAC = 0.5  
AUGMENT_RC = True 
AUGMENT_OFFSET_MAX_BP = 200 
MAX_PEAKS_PER_MARK_PER_CELL = 2500 # Reduced from 20k for 4x speedup

# Loss weighting: alpha * Huber + (1-alpha) * Pearson
LOSS_ALPHA = 0.2 

# Presence/absence threshold in log2(signal + 1) space for derived metrics
PRESENCE_THRESHOLD = 0.5

# Genome Cache
GENOME_CACHE_PATH = os.path.join(PROCESSED_DIR, "genome_hg38.bin")

# Dataloader
NUM_WORKERS = 4 # Slightly reduced to avoid worker overhead on some systems
PIN_MEMORY = True
