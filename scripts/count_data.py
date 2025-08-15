import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
from config import config
import torch

print(config.NUM_TRAINING_STEPS) 
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Current usage: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Max usage: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")



def count_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        print(f"X File not found: {csv_path}")
        return 0
    
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)  # Skip header
        for row in r:
            if row and len(row) >= 2:  # Valid row with path and text
                count += 1
    return count

# Count both datasets
train_csv = config.TRAIN_CSV
test_csv = os.path.join(config.FILELIST_PATH, "test", "test.csv")

print("=== Data Summary ===")
train_count = count_csv_rows(train_csv)
test_count = count_csv_rows(test_csv)

print(f"Training samples: {train_count}")
print(f"Test samples: {test_count}")
print(f"Total samples: {train_count + test_count}")

if train_count > 0:
    print(f"\Recommended training settings:")
    print(f"   N_SAMPLES: {min(train_count, 500)}")  # Cap at 500 for reasonable training time
    print(f"   NUM_TRAINING_STEPS: {max(50, min(train_count // 10, 200))}")  # Scale with data size