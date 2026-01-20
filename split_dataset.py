"""
Dataset Splitter - Crea train/test split del dataset JSONL
Estratificado por tipo de pregunta (A/B/C) si es posible
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Config
INPUT_FILE = r"d:\FineTuning\dataset\train.jsonl"
TRAIN_FILE = r"d:\FineTuning\dataset\train_split.jsonl"
TEST_FILE = r"d:\FineTuning\dataset\test_split.jsonl"
TEST_SIZE = 0.20  # 20% para test
RANDOM_SEED = 42

def detect_question_type(text):
    """Detecta el tipo de pregunta (A, B, o C) basado en el system prompt"""
    if "precisión normativa" in text:
        return "A"  # Formal
    elif "Explicas la normativa" in text:
        return "B"  # Temático
    elif "compañero estudiante" in text or "empatía" in text:
        return "C"  # Humanizado
    return "Unknown"

def main():
    random.seed(RANDOM_SEED)
    
    # Leer dataset completo
    print(f"📂 Leyendo dataset: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    total_samples = len(all_data)
    print(f"✅ Total de samples: {total_samples}")
    
    # Agrupar por tipo (A/B/C) para split estratificado
    grouped = defaultdict(list)
    for sample in all_data:
        q_type = detect_question_type(sample['text'])
        grouped[q_type].append(sample)
    
    print(f"\n📊 Distribución por tipo:")
    for q_type, samples in grouped.items():
        print(f"   Tipo {q_type}: {len(samples)} samples")
    
    # Split estratificado
    train_data = []
    test_data = []
    
    for q_type, samples in grouped.items():
        random.shuffle(samples)
        n_test = max(1, int(len(samples) * TEST_SIZE))  # Al menos 1 por tipo
        
        test_data.extend(samples[:n_test])
        train_data.extend(samples[n_test:])
    
    # Shuffle final
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Guardar splits
    print(f"\n💾 Guardando splits:")
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"   Train: {len(train_data)} samples → {TRAIN_FILE}")
    
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"   Test:  {len(test_data)} samples → {TEST_FILE}")
    
    # Verificar distribución en splits
    print(f"\n✅ Split completado:")
    print(f"   Train: {len(train_data)}/{total_samples} ({len(train_data)/total_samples*100:.1f}%)")
    print(f"   Test:  {len(test_data)}/{total_samples} ({len(test_data)/total_samples*100:.1f}%)")
    
    # Distribución por tipo en cada split
    train_types = defaultdict(int)
    test_types = defaultdict(int)
    
    for sample in train_data:
        train_types[detect_question_type(sample['text'])] += 1
    for sample in test_data:
        test_types[detect_question_type(sample['text'])] += 1
    
    print(f"\n📊 Distribución Train:")
    for q_type, count in train_types.items():
        print(f"   Tipo {q_type}: {count}")
    
    print(f"\n📊 Distribución Test:")
    for q_type, count in test_types.items():
        print(f"   Tipo {q_type}: {count}")

if __name__ == "__main__":
    main()
