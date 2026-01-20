"""
Split Dataset V3 - Mantiene compatibilidad con V3
"""

import json
import random
from pathlib import Path

# Config
INPUT_FILE = r"d:\FineTuning\dataset\train_v3.jsonl"
TRAIN_FILE = r"d:\FineTuning\dataset\train_v3_split.jsonl"
TEST_FILE = r"d:\FineTuning\dataset\test_v3_split.jsonl"
TEST_SIZE = 0.15  # 15% para test
RANDOM_SEED = 42

def main():
    random.seed(RANDOM_SEED)
    
    print(f"📂 Leyendo: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    total = len(all_data)
    print(f"✅ Total: {total} samples")
    
    random.shuffle(all_data)
    
    n_test = int(len(all_data) * TEST_SIZE)
    test_data = all_data[:n_test]
    train_data = all_data[n_test:]
    
    print(f"\n💾 Guardando splits:")
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"   Train: {len(train_data)} → {TRAIN_FILE}")
    
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"   Test:  {len(test_data)} → {TEST_FILE}")
    
    print(f"\n✅ Split V3 completado ({len(train_data)}/{len(test_data)})")

if __name__ == "__main__":
    main()
