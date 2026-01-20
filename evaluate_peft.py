"""
Script de Evaluación para Modelo PEFT/LoRA (Compatible Windows)
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate
from tqdm import tqdm
import json
import os

# Configuración
MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
PEFT_MODEL_ID = r"d:\FineTuning\outputs_peft\final_model"
TEST_FILE = r"d:\FineTuning\dataset\test_split.jsonl"
OUTPUT_FILE = r"d:\FineTuning\outputs_peft\evaluation_results.json"

def main():
    print(f"🚀 Iniciando evaluación...")
    
    # 1. Cargar Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Cargar Modelo Base (4-bit)
    print("📦 Cargando modelo base...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    # 3. Cargar Adaptador LoRA
    print(f"🔗 Cargando adaptador LoRA desde {PEFT_MODEL_ID}...")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    model.eval()

    # 4. Cargar Métricas
    print("📏 Cargando métricas...")
    bertscore = evaluate.load("bertscore")
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    # 5. Cargar Datos de Test
    dataset = load_dataset("json", data_files=TEST_FILE, split="train")
    print(f"📊 Evaluando sobre {len(dataset)} ejemplos...")

    references = []
    predictions = []
    
    # Procesar ejemplos
    for item in tqdm(dataset):
        # El dataset tiene formato chat: [{"role": "user", ...}, {"role": "assistant", ...}]
        # Extraer prompt (user) y referencia (assistant)
        messages = item["text"] # Asumiendo formato chat template directo en texto
        # Si 'text' es string raw del chat template:
        # User dice X, assistant dice Y.
        # Necesitamos extraer la parte del usuario para predecir.
        
        # HACK: Para simplicidad, usaremos el mismo pipeline de generación
        # Cortar el prompt hasta antes de la respuesta del asistente
        prompt = item["text"].split("<|start_header_id|>assistant<|end_header_id|>")[0] + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Referencia real (lo que sigue)
        try:
            reference = item["text"].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1].replace("<|eot_id|>", "").strip()
        except IndexError:
            continue # Skip malformed

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraer solo la respuesta nueva (quitando el prompt)
        # Como decode quita special tokens, el split por header id es difícil.
        # Mejor estrategia: decode(outputs[0][input_len:])
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        predictions.append(response)
        references.append(reference)

    # 6. Calcular Métricas
    print("\n📈 Calculando puntajes...")
    
    # BERTScore
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="es", verbose=True)
    f1_mean = np.mean(bert_results['f1'])
    
    # BLEU
    bleu_results = bleu.compute(predictions=predictions, references=[[r] for r in references])
    
    # ROUGE
    rouge_results = rouge.compute(predictions=predictions, references=references)
    
    results = {
        "bertscore_f1": float(f1_mean),
        "bleu": bleu_results["score"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "samples": len(predictions)
    }
    
    print("\n" + "="*50)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*50)
    print(json.dumps(results, indent=2))
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
