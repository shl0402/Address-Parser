import os
import json
import random

def merge_and_split_data(files_to_be_added, output_dir="data", split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Reads a list of JSONL files, shuffles the combined data, splits it, 
    and appends to the target train/val/test files.
    """
    # 1. Define output file paths
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "validation.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")
    
    # 2. Read and aggregate all new data
    all_new_data = []
    for file_path in files_to_be_added:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found and skipped -> {file_path}")
            continue
            
        print(f"📖 Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Validate it's a proper JSON object before adding
                try:
                    json.loads(line)
                    all_new_data.append(line)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON line in {file_path}.")
                        
    if not all_new_data:
        print("❌ No valid data found in the provided files.")
        return
        
    print(f"✅ Loaded {len(all_new_data)} total valid lines. Shuffling data...")
    
    # 3. Shuffle for random distribution
    random.seed(seed)
    random.shuffle(all_new_data)
    
    # 4. Calculate split indices
    total = len(all_new_data)
    train_idx = int(total * split_ratios[0])
    val_idx = int(total * (split_ratios[0] + split_ratios[1]))
    
    train_split = all_new_data[:train_idx]
    val_split = all_new_data[train_idx:val_idx]
    test_split = all_new_data[val_idx:]
    
    # 5. Append to target files ('a' mode creates the file if it doesn't exist)
    def append_to_file(data, filename):
        if not data:
            return
        with open(filename, 'a', encoding='utf-8') as f:
            for line in data:
                f.write(line + "\n")
                
    print(f"📝 Appending {len(train_split)} lines to {train_file}")
    append_to_file(train_split, train_file)
    
    print(f"📝 Appending {len(val_split)} lines to {val_file}")
    append_to_file(val_split, val_file)
    
    print(f"📝 Appending {len(test_split)} lines to {test_file}")
    append_to_file(test_split, test_file)
    
    print("🎉 Merge and split complete!")

if __name__ == "__main__":
    # --- Configuration ---
    
    # Put the paths to your new files here
    files_to_be_added = [
        "auto_train_villageV3.jsonl",
        "compact_train.jsonl",
        "compact_test.jsonl",
        "compact_validation.jsonl"
    ]
    
    # The directory where train.jsonl, validation.jsonl, etc., will live
    target_directory = "data" 
    
    # Run the merger
    merge_and_split_data(files_to_be_added, output_dir=target_directory)