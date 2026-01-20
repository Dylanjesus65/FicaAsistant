"""
Evaluación de Métricas para Modelos Fine-Tuned
BLEU, ROUGE, Perplexity, BERTScore
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# Métricas
from evaluate import load as load_metric

# Configuración
MODEL_BASE = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_PATH = r"d:\FineTuning\chatbot\chatbot_app\Llama-3.2-3B-entrenado-v3"
TEST_FILE = r"d:\FineTuning\dataset\test_v3_split.jsonl"
OUTPUT_FILE = r"d:\FineTuning\outputs_peft\metrics_v3.json"

SYSTEM_PROMPT = "Eres FicaAsistant de la UTN. Responde solo con información de la normativa oficial. Si no sabes, di: 'No tengo esa información en la normativa.'"

def load_model():
    print("🔄 Cargando modelo...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        quantization_config=bnb_config,
        device_map="cuda",
        torch_dtype=torch.float16
    )
    
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ Modelo cargado")
    
    return model, tokenizer

def extract_qa_from_text(text):
    """Extrae pregunta y respuesta esperada del formato de entrenamiento"""
    try:
        # Buscar user y assistant
        user_start = text.find("<|start_header_id|>user<|end_header_id|>") + len("<|start_header_id|>user<|end_header_id|>")
        user_end = text.find("<|eot_id|>", user_start)
        question = text[user_start:user_end].strip()
        
        assistant_start = text.find("<|start_header_id|>assistant<|end_header_id|>") + len("<|start_header_id|>assistant<|end_header_id|>")
        assistant_end = text.find("<|eot_id|>", assistant_start)
        reference = text[assistant_start:assistant_end].strip()
        
        return question, reference
    except:
        return None, None

def generate_response(model, tokenizer, question, max_tokens=200):
    """Genera respuesta del modelo"""
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,  # Bajo para respuestas más determinísticas
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()

def calculate_perplexity(model, tokenizer, texts, max_samples=50):
    """Calcula perplexity promedio"""
    print("📊 Calculando Perplexity...")
    perplexities = []
    
    for text in tqdm(texts[:max_samples], desc="PPL"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            ppl = np.exp(loss)
            if ppl < 1000:  # Filtrar valores extremos
                perplexities.append(ppl)
    
    return np.mean(perplexities) if perplexities else float('inf')

def main():
    print("🚀 Evaluación de Métricas V3")
    
    # Cargar modelo
    model, tokenizer = load_model()
    
    # Cargar datos de prueba
    dataset = load_dataset('json', data_files=TEST_FILE, split='train')
    print(f"📂 Ejemplos de prueba: {len(dataset)}")
    
    # Extraer QA pairs
    qa_pairs = []
    for item in dataset:
        q, r = extract_qa_from_text(item['text'])
        if q and r:
            qa_pairs.append((q, r))
    
    print(f"✅ Pares Q/A válidos: {len(qa_pairs)}")
    
    # Generar respuestas
    predictions = []
    references = []
    
    print("\n📝 Generando respuestas...")
    for question, reference in tqdm(qa_pairs[:30], desc="Generando"):  # Limitar a 30
        prediction = generate_response(model, tokenizer, question)
        predictions.append(prediction)
        references.append(reference)
    
    # Calcular métricas
    results = {}
    
    # 1. BLEU
    print("\n📊 Calculando BLEU...")
    bleu = load_metric("bleu")
    bleu_result = bleu.compute(
        predictions=[p.split() for p in predictions],
        references=[[r.split()] for r in references]
    )
    results["bleu"] = round(bleu_result["bleu"] * 100, 2)
    
    # 2. ROUGE
    print("📊 Calculando ROUGE...")
    rouge = load_metric("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=references)
    results["rouge1"] = round(rouge_result["rouge1"] * 100, 2)
    results["rouge2"] = round(rouge_result["rouge2"] * 100, 2)
    results["rougeL"] = round(rouge_result["rougeL"] * 100, 2)
    
    # 3. Perplexity
    all_texts = [item['text'] for item in dataset]
    results["perplexity"] = round(calculate_perplexity(model, tokenizer, all_texts), 2)
    
    # 4. BERTScore
    print("📊 Calculando BERTScore...")
    try:
        bertscore = load_metric("bertscore")
        bert_result = bertscore.compute(predictions=predictions, references=references, lang="es")
        results["bertscore_precision"] = round(np.mean(bert_result["precision"]) * 100, 2)
        results["bertscore_recall"] = round(np.mean(bert_result["recall"]) * 100, 2)
        results["bertscore_f1"] = round(np.mean(bert_result["f1"]) * 100, 2)
    except Exception as e:
        print(f"⚠️ Error en BERTScore: {e}")
        results["bertscore_f1"] = "N/A"
    
    # Guardar resultados
    print("\n" + "="*50)
    print("📊 RESULTADOS DE MÉTRICAS V3")
    print("="*50)
    for metric, value in results.items():
        print(f"   {metric}: {value}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Resultados guardados en: {OUTPUT_FILE}")
    
    # Mostrar ejemplos
    print("\n📝 EJEMPLOS DE RESPUESTAS:")
    print("-"*50)
    for i in range(min(3, len(predictions))):
        print(f"Q: {qa_pairs[i][0][:80]}...")
        print(f"Esperado: {references[i][:100]}...")
        print(f"Generado: {predictions[i][:100]}...")
        print("-"*50)

if __name__ == "__main__":
    main()
