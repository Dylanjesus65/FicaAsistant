"""
Fine-Tuning con Unsloth + QLoRA
Llama-3.2-3B-Instruct para Asistente Virtual UTN-FICA

Optimizado para RTX 2070 SUPER (8GB VRAM)
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
import evaluate

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Paths
TRAIN_FILE = r"d:\FineTuning\dataset\train_split.jsonl"
TEST_FILE = r"d:\FineTuning\dataset\test_split.jsonl"
OUTPUT_DIR = r"d:\FineTuning\outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Model
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 1024

# QLoRA Config (Optimizado para 8GB VRAM)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 8
PER_DEVICE_EVAL_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Logging & Checkpointing
LOGGING_STEPS = 5
EVAL_STEPS = 25
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 3

# Crear directorios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# CARGAR MODELO CON UNSLOTH
# ============================================================================

print("🚀 Cargando modelo con Unsloth...")
print(f"   Modelo: {MODEL_NAME}")
print(f"   Cuantización: 4-bit (QLoRA)")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (bfloat16 para RTX 2070 SUPER)
    load_in_4bit=True,  # QLoRA 4-bit quantization
)

# Configurar LoRA
print(f"\n⚙️ Configurando LoRA:")
print(f"   Rank (r): {LORA_R}")
print(f"   Alpha: {LORA_ALPHA}")
print(f"   Dropout: {LORA_DROPOUT}")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth optimizado
    random_state=42,
)

# ============================================================================
# CARGAR DATASETS
# ============================================================================

print(f"\n📂 Cargando datasets:")
print(f"   Train: {TRAIN_FILE}")
print(f"   Test:  {TEST_FILE}")

train_dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')
eval_dataset = load_dataset('json', data_files=TEST_FILE, split='train')

print(f"✅ Train samples: {len(train_dataset)}")
print(f"✅ Test samples:  {len(eval_dataset)}")

# ============================================================================
# FORMATEO DE DATOS
# ============================================================================

def formatting_prompts_func(examples):
    """
    Formatea los samples al formato esperado por el modelo.
    El dataset ya viene con el formato Llama-3 completo en 'text'.
    """
    texts = examples["text"]
    return {"text": texts}

# ============================================================================
# CALLBACKS PARA VISUALIZACIÓN
# ============================================================================

class LossPlotCallback(TrainerCallback):
    """Callback para graficar pérdida en tiempo real"""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        self.perplexities = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.train_steps.append(state.global_step)
            
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
                # Calcular perplexity = exp(loss)
                perplexity = np.exp(logs["eval_loss"])
                self.perplexities.append(perplexity)
                print(f"\n📊 Eval Metrics (Step {state.global_step}):")
                print(f"   Loss: {logs['eval_loss']:.4f}")
                print(f"   Perplexity: {perplexity:.2f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generar gráficos al finalizar"""
        print("\n📈 Generando gráficos de pérdida...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Training Loss
        axes[0].plot(self.train_steps, self.train_losses, 
                     label="Train Loss", color="blue", linewidth=2)
        axes[0].set_xlabel("Steps", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Training Loss", fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Evaluation Loss
        if self.eval_losses:
            axes[1].plot(self.eval_steps, self.eval_losses, 
                         label="Eval Loss", color="orange", linewidth=2, marker='o')
            axes[1].set_xlabel("Steps", fontsize=12)
            axes[1].set_ylabel("Loss", fontsize=12)
            axes[1].set_title("Evaluation Loss", fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. Perplexity
        if self.perplexities:
            axes[2].plot(self.eval_steps, self.perplexities, 
                         label="Perplexity", color="green", linewidth=2, marker='s')
            axes[2].axhline(y=20, color='red', linestyle='--', 
                            label='Meta (<20)', alpha=0.7)
            axes[2].set_xlabel("Steps", fontsize=12)
            axes[2].set_ylabel("Perplexity", fontsize=12)
            axes[2].set_title("Perplexity (Lower = Better)", 
                              fontsize=14, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {plot_path}")
        plt.close()
        
        # Guardar métricas en JSON
        metrics = {
            "train_losses": [float(x) for x in self.train_losses],
            "eval_losses": [float(x) for x in self.eval_losses],
            "perplexities": [float(x) for x in self.perplexities],
            "train_steps": self.train_steps,
            "eval_steps": self.eval_steps,
            "final_train_loss": float(self.train_losses[-1]) if self.train_losses else None,
            "final_eval_loss": float(self.eval_losses[-1]) if self.eval_losses else None,
            "final_perplexity": float(self.perplexities[-1]) if self.perplexities else None
        }
        
        metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Métricas guardadas: {metrics_path}")

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

print(f"\n⚙️ Configurando entrenamiento:")
print(f"   Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE} (effective: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {NUM_TRAIN_EPOCHS}")
print(f"   Max seq length: {MAX_SEQ_LENGTH}")

training_args = TrainingArguments(
    # Output
    output_dir=CHECKPOINT_DIR,
    
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    
    # Optimization
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    optim="paged_adamw_8bit",  # 8-bit optimizer (ahorra VRAM)
    
    # Precision
    fp16=True,   # RTX 2070 SUPER usa fp16 (Turing)
    bf16=False,  # bfloat16 requiere Ampere (30xx) o superior
    
    # Logging & Evaluation
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Performance
    group_by_length=True,
    dataloader_num_workers=4,  # i7-9700 tiene 8 cores
    
    # Misc
    report_to="none",  # Deshabilitar wandb/tensorboard
    seed=42,
)

# ============================================================================
# TRAINER
# ============================================================================

print("\n🎯 Inicializando Trainer...")

loss_callback = LossPlotCallback()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    formatting_func=formatting_prompts_func,
    args=training_args,
    callbacks=[loss_callback],
)

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

print("\n" + "="*70)
print("🚀 INICIANDO FINE-TUNING")
print("="*70)
print(f"Configuración:")
print(f"  • Modelo: {MODEL_NAME}")
print(f"  • Train samples: {len(train_dataset)}")
print(f"  • Test samples: {len(eval_dataset)}")
print(f"  • Epochs: {NUM_TRAIN_EPOCHS}")
print(f"  • Effective batch: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  • Estimado: ~40 minutos")
print("="*70 + "\n")

# Mostrar uso de GPU antes de empezar
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   VRAM Asignada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   VRAM Reservada: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB\n")

# ENTRENAR
trainer.train()

print("\n" + "="*70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*70)

# ============================================================================
# GUARDAR MODELO FINAL
# ============================================================================

print("\n💾 Guardando modelo final...")

final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"✅ Modelo guardado en: {final_model_dir}")

# Guardar también en formato GGUF (opcional, para inferencia rápida)
print("\n📦 Guardando modelo en formato GGUF (para inferencia)...")
try:
    model.save_pretrained_gguf(
        os.path.join(OUTPUT_DIR, "final_model_gguf"),
        tokenizer,
        quantization_method="q4_k_m"  # 4-bit cuantización
    )
    print("✅ Modelo GGUF guardado")
except Exception as e:
    print(f"⚠️ No se pudo guardar GGUF: {e}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print("📊 RESUMEN DEL ENTRENAMIENTO")
print("="*70)

if hasattr(trainer.state, 'best_metric'):
    print(f"✅ Mejor Eval Loss: {trainer.state.best_metric:.4f}")

if loss_callback.perplexities:
    final_perplexity = loss_callback.perplexities[-1]
    print(f"✅ Perplexity Final: {final_perplexity:.2f}")
    
    if final_perplexity < 15:
        print("   🌟 EXCELENTE - Modelo muy bien entrenado!")
    elif final_perplexity < 25:
        print("   ✅ BUENO - Modelo funcionando correctamente")
    else:
        print("   ⚠️ REGULAR - Considerar más epochs o ajustar hiperparámetros")

print(f"\n📁 Archivos generados:")
print(f"   • Modelo final: {final_model_dir}")
print(f"   • Checkpoints: {CHECKPOINT_DIR}")
print(f"   • Gráficos: {os.path.join(OUTPUT_DIR, 'loss_curves.png')}")
print(f"   • Métricas: {os.path.join(OUTPUT_DIR, 'training_metrics.json')}")

print("\n🎉 ¡Proceso completado! Ejecuta 'evaluate_model.py' para métricas detalladas.")
print("="*70 + "\n")
