import os, csv

# ====== PATH CONFIGURATION ======
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
#print("ROOT_DIR is:", ROOT_DIR)

TRAIN_PATH = os.path.join(ROOT_DIR, "data", "data", "train")
TEST_PATH  = os.path.join(ROOT_DIR, "data", "data", "test")
FILELIST_PATH = os.path.join(ROOT_DIR, "filelists")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "processed_data")

# ====== PROCESSING PARAMETERS ======
# Train
TRAIN_PAD = 0.5
TRAIN_VOLUME = 2.0

# Test
TEST_PAD = 0.0
TEST_VOLUME = 1.0

EVAL_MODEL_PATH = "openai/whisper-base"  # Add a comment to the front of this line if you are switching to the fine tuned agent below
#EVAL_MODEL_PATH = "./fine_tuned_whisper"  #Switch to this for fine-tuned
EVAL_SAMPLES = -1  # -1 = all test samples, or set a number


# Training:

TRAIN_CSV = os.path.join(FILELIST_PATH, "train", "train.csv")
TEST_CSV = os.path.join(FILELIST_PATH, "test", "test.csv")
MODEL_ID = "openai/whisper-base"
N_SAMPLES = 6923  #Number of training samples (6923 = full dataset)
#N_SAMPLES = 1000 (For testing and smaller GPU's)

#Training settings
LEARNING_RATE = 1e-4  #Learning rate for fine-tuning
NUM_TRAINING_STEPS = 13846 #Training iterations
PRINT_EVERY_X_STEPS = 100  #Progress update frequency 

#LoRA settings  
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

