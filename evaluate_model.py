"""
Evaluación Detallada del Modelo Fine-Tuneado
Métricas: BERTScore, BLEU, ROUGE, Perplexity
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Paths
MODEL_DIR = r"d:\FineTuning\outputs\final_model"
TEST_FILE = r"d:\FineTuning\dataset\test_split.jsonl"
OUTPUT_DIR = r"d:\FineTuning\outputs"
EVAL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "evaluation_results.json")

# Generation Config
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

print("="*70)
print("🔬 EVALUACIÓN DETALLADA DEL MODELO")
print("="*70)

# ============================================================================
# CARGAR MODELO Y TOKENIZER
# ============================================================================

print(f"\n📂 Cargando modelo desde: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print(f"✅ Modelo cargado")
print(f"   Parámetros: {model.num_parameters() / 1e9:.2f}B")

# ============================================================================
# CARGAR TEST DATASET
# ============================================================================

print(f"\n📊 Cargando test dataset: {TEST_FILE}")
test_dataset = load_dataset('json', data_files=TEST_FILE, split='train')
print(f"✅ Test samples: {len(test_dataset)}")

# ============================================================================
# CARGAR MÉTRICAS
# ============================================================================

print(f"\n📈 Cargando métricas de evaluación...")

bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

print("✅ Métricas cargadas: BERTScore, BLEU, ROUGE, Perplexity")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def extract_user_question(text):
    """Extrae la pregunta del usuario del formato Llama-3"""
    try:
        user_start = text.find("<|start_header_id|>user<|end_header_id|>")
        if user_start == -1:
            return None
        
        user_start = user_start + len("<|start_header_id|>user<|end_header_id|>")
        user_end = text.find("<|eot_id|>", user_start)
        
        if user_end == -1:
            return None
        
        question = text[user_start:user_end].strip()
        return question
    except:
        return None

def extract_reference_answer(text):
    """Extrae la respuesta de referencia del formato Llama-3"""
    try:
        assistant_start = text.find("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start == -1:
            return None
        
        assistant_start = assistant_start + len("<|start_header_id|>assistant<|end_header_id|>")
        assistant_end = text.find("<|eot_id|>", assistant_start)
        
        if assistant_end == -1:
            # Tomar hasta el final
            answer = text[assistant_start:].strip()
        else:
            answer = text[assistant_start:assistant_end].strip()
        
        return answer
    except:
        return None

def extract_system_prompt(text):
    """Extrae el system prompt"""
    try:
        system_start = text.find("<|start_header_id|>system<|end_header_id|>")
        if system_start == -1:
            return ""
        
        system_start = system_start + len("<|start_header_id|>system<|end_header_id|>")
        system_end = text.find("<|eot_id|>", system_start)
        
        if system_end == -1:
            return ""
        
        system_prompt = text[system_start:system_end].strip()
        return system_prompt
    except:
        return ""

def generate_answer(system_prompt, question):
    """Genera respuesta del modelo"""
    # Construir prompt
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar solo la parte generada
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                       skip_special_tokens=True)
    
    return generated_text.strip()

# ============================================================================
# EVALUACIÓN
# ============================================================================

print(f"\n🔄 Generando respuestas del modelo...")
print(f"   (Esto puede tomar ~5-10 minutos para {len(test_dataset)} samples)")

predictions = []
references = []
systems = []
questions_list = []
perplexities = []

for i, sample in enumerate(tqdm(test_dataset, desc="Evaluando")):
    text = sample['text']
    
    # Extraer componentes
    system_prompt = extract_system_prompt(text)
    question = extract_user_question(text)
    reference = extract_reference_answer(text)
    
    if question is None or reference is None:
        print(f"⚠️ Saltando sample {i}: formato inválido")
        continue
    
    # Generar respuesta
    try:
        prediction = generate_answer(system_prompt, question)
        
        predictions.append(prediction)
        references.append(reference)
        systems.append(system_prompt)
        questions_list.append(question)
        
        # Calcular perplexity para esta respuesta
        # Construir input completo
        full_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{reference}<|eot_id|>"
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            perplexity = np.exp(loss)
            perplexities.append(perplexity)
            
    except Exception as e:
        print(f"⚠️ Error en sample {i}: {e}")
        continue

print(f"\n✅ Generación completada: {len(predictions)} predicciones")

# ============================================================================
# CALCULAR MÉTRICAS
# ============================================================================

print(f"\n📊 Calculando métricas...")

# BERTScore
print("   • BERTScore...")
bert_results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="es",
    model_type="bert-base-multilingual-cased"
)

# BLEU
print("   • BLEU...")
bleu_results = bleu.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

# ROUGE
print("   • ROUGE...")
rouge_results = rouge.compute(
    predictions=predictions,
    references=references
)

# Perplexity promedio
avg_perplexity = np.mean(perplexities)

# ============================================================================
# RESULTADOS
# ============================================================================

results = {
    "model_path": MODEL_DIR,
    "test_samples": len(predictions),
    "metrics": {
        "bertscore": {
            "precision": float(np.mean(bert_results['precision'])),
            "recall": float(np.mean(bert_results['recall'])),
            "f1": float(np.mean(bert_results['f1']))
        },
        "bleu": float(bleu_results['bleu']),
        "rouge": {
            "rouge1": float(rouge_results['rouge1']),
            "rouge2": float(rouge_results['rouge2']),
            "rougeL": float(rouge_results['rougeL']),
            "rougeLsum": float(rouge_results['rougeLsum'])
        },
        "perplexity": {
            "mean": float(avg_perplexity),
            "std": float(np.std(perplexities)),
            "min": float(np.min(perplexities)),
            "max": float(np.max(perplexities))
        }
    },
    "per_sample_perplexity": [float(p) for p in perplexities]
}

# Guardar resultados
with open(EVAL_RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Resultados guardados en: {EVAL_RESULTS_FILE}")

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

print(f"\n📈 Generando visualizaciones...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. BERTScore (Precision, Recall, F1)
ax1 = axes[0, 0]
bert_metrics = ['Precision', 'Recall', 'F1']
bert_values = [
    results['metrics']['bertscore']['precision'],
    results['metrics']['bertscore']['recall'],
    results['metrics']['bertscore']['f1']
]
bars1 = ax1.bar(bert_metrics, bert_values, color=['#3498db', '#2ecc71', '#e74c3c'])
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('BERTScore Metrics', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, bert_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. ROUGE Metrics
ax2 = axes[0, 1]
rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
rouge_values = [
    results['metrics']['rouge']['rouge1'],
    results['metrics']['rouge']['rouge2'],
    results['metrics']['rouge']['rougeL']
]
bars2 = ax2.bar(rouge_metrics, rouge_values, color=['#9b59b6', '#f39c12', '#1abc9c'])
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('ROUGE Metrics', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, rouge_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. BLEU + Perplexity
ax3 = axes[1, 0]
combined_metrics = ['BLEU']
combined_values = [results['metrics']['bleu']]
bars3 = ax3.bar(combined_metrics, combined_values, color='#e67e22', width=0.4)
ax3.set_ylabel('BLEU Score', fontsize=11)
ax3.set_title('BLEU Score', fontsize=13, fontweight='bold')
ax3.set_ylim(0, 1)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars3, combined_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Agregar texto de perplexity
perp_text = f"Perplexity:\n{avg_perplexity:.2f}"
ax3.text(0.5, 0.5, perp_text, transform=ax3.transAxes, 
         fontsize=14, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Distribución de Perplexity
ax4 = axes[1, 1]
ax4.hist(perplexities, bins=15, color='#16a085', edgecolor='black', alpha=0.7)
ax4.axvline(avg_perplexity, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_perplexity:.2f}')
ax4.axvline(20, color='orange', linestyle=':', linewidth=2, label='Target: <20')
ax4.set_xlabel('Perplexity', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Perplexity Distribution', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {plot_path}")
plt.close()

# ============================================================================
# REPORTE FINAL
# ============================================================================

print("\n" + "="*70)
print("📊 RESULTADOS DE EVALUACIÓN")
print("="*70)

print(f"\n🎯 BERTScore:")
print(f"   • Precision: {results['metrics']['bertscore']['precision']:.4f}")
print(f"   • Recall:    {results['metrics']['bertscore']['recall']:.4f}")
print(f"   • F1:        {results['metrics']['bertscore']['f1']:.4f}")

print(f"\n📝 BLEU:")
print(f"   • Score: {results['metrics']['bleu']:.4f}")

print(f"\n📄 ROUGE:")
print(f"   • ROUGE-1: {results['metrics']['rouge']['rouge1']:.4f}")
print(f"   • ROUGE-2: {results['metrics']['rouge']['rouge2']:.4f}")
print(f"   • ROUGE-L: {results['metrics']['rouge']['rougeL']:.4f}")

print(f"\n🎲 Perplexity:")
print(f"   • Mean: {avg_perplexity:.2f}")
print(f"   • Std:  {np.std(perplexities):.2f}")
print(f"   • Min:  {np.min(perplexities):.2f}")
print(f"   • Max:  {np.max(perplexities):.2f}")

# Evaluación cualitativa
print(f"\n✨ Evaluación Cualitativa:")

# BERTScore F1
if results['metrics']['bertscore']['f1'] > 0.88:
    print("   • BERTScore: 🌟 EXCELENTE - Alta similitud semántica")
elif results['metrics']['bertscore']['f1'] > 0.82:
    print("   • BERTScore: ✅ BUENO - Similitud semántica adecuada")
else:
    print("   • BERTScore: ⚠️ MEJORABLE - Considerar más entrenamiento")

# BLEU
if results['metrics']['bleu'] > 0.50:
    print("   • BLEU: 🌟 EXCELENTE - Alta precisión de n-gramas")
elif results['metrics']['bleu'] > 0.35:
    print("   • BLEU: ✅ BUENO - Precisión aceptable")
else:
    print("   • BLEU: ⚠️ MEJORABLE - Respuestas poco literales")

# Perplexity
if avg_perplexity < 15:
    print("   • Perplexity: 🌟 EXCELENTE - Modelo muy confiado")
elif avg_perplexity < 25:
    print("   • Perplexity: ✅ BUENO - Modelo confiado")
else:
    print("   • Perplexity: ⚠️ MEJORABLE - Modelo poco confiado")

print("\n" + "="*70)
print(f"📁 Archivos generados:")
print(f"   • Resultados JSON: {EVAL_RESULTS_FILE}")
print(f"   • Gráficos: {plot_path}")
print("="*70 + "\n")

# Guardar ejemplos de predicciones
examples_file = os.path.join(OUTPUT_DIR, "prediction_examples.txt")
with open(examples_file, 'w', encoding='utf-8') as f:
    f.write("EJEMPLOS DE PREDICCIONES\n")
    f.write("="*70 + "\n\n")
    
    for i in range(min(5, len(predictions))):
        f.write(f"EJEMPLO {i+1}:\n")
        f.write(f"System: {systems[i]}\n\n")
        f.write(f"Pregunta: {questions_list[i]}\n\n")
        f.write(f"Referencia:\n{references[i]}\n\n")
        f.write(f"Predicción:\n{predictions[i]}\n\n")
        f.write(f"Perplexity: {perplexities[i]:.2f}\n")
        f.write("-"*70 + "\n\n")

print(f"✅ Ejemplos guardados: {examples_file}")
print(f"\n🎉 Evaluación completada!\n")
