"""
Fine-Tuning con PEFT + QLoRA (Versión Compatible Windows)
Llama-3.2-3B-Instruct para Asistente Virtual UTN-FICA

Optimizado para RTX 2070 SUPER (8GB VRAM) usando librerías estándar de HuggingFace.
"""

import os
import json
import torch
torch.cuda.empty_cache() # Limpiar VRAM al inicio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    TrainerCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType
)
from trl import SFTTrainer
import evaluate
import shutil

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Paths
TRAIN_FILE = r"d:\FineTuning\dataset\train_split.jsonl"
TEST_FILE = r"d:\FineTuning\dataset\test_split.jsonl"
OUTPUT_DIR = r"d:\FineTuning\outputs_peft"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Model
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct" # Usamos el mismo modelo base
MAX_SEQ_LENGTH = 1024

# QLoRA Config (Optimizado para 8GB VRAM)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
PER_DEVICE_EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_STEPS = 600  # <--- Changed to 600 steps for better convergence
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Logging & Checkpointing
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STRATEGY = "no" # Don't save intermediate checkpoints to save space/time

# Crear directorios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# CARGAR MODELO (Estilo HF Nativo)
# ============================================================================

print("🚀 Cargando modelo con PEFT/BitsAndBytes...")
print(f"   Modelo: {MODEL_NAME}")
print(f"   Cuantización: 4-bit (QLoRA)")

# Configuración de cuantización 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # Turing usa fp16
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},  # FIX: Forzar mapeo a GPU 0 evita error .to() en Windows
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Preparar modelo para k-bit training
model = prepare_model_for_kbit_training(model)

# Configurar LoRA
print(f"\n⚙️ Configurando LoRA:")
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

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
    return {"text": examples["text"]}

# ============================================================================
# CALLBACKS PARA VISUALIZACIÓN (Reutilizado)
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
                try:
                    perplexity = np.exp(logs["eval_loss"])
                except OverflowError:
                    perplexity = float('inf')
                self.perplexities.append(perplexity)
                print(f"\n📊 Eval Metrics (Step {state.global_step}):")
                print(f"   Loss: {logs['eval_loss']:.4f}")
                print(f"   Perplexity: {perplexity:.2f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        print("\n📈 Generando gráficos de pérdida...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Training Loss
        if self.train_steps:
            axes[0].plot(self.train_steps, self.train_losses, label="Train Loss", color="blue")
            axes[0].set_title("Training Loss")
            axes[0].legend()
            axes[0].grid(True)
        
        # 2. Evaluation Loss
        if self.eval_steps:
            axes[1].plot(self.eval_steps, self.eval_losses, label="Eval Loss", color="orange")
            axes[1].set_title("Evaluation Loss")
            axes[1].legend()
            axes[1].grid(True)
            
        # 3. Perplexity
        if self.eval_steps:
            axes[2].plot(self.eval_steps, self.perplexities, label="Perplexity", color="green")
            axes[2].set_title("Perplexity")
            axes[2].legend()
            axes[2].grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"))
        plt.close()

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_steps=MAX_STEPS, # Using max_steps
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    optim="paged_adamw_8bit",
    fp16=True, # RTX 2070 Turing
    bf16=False,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy=SAVE_STRATEGY,
    load_best_model_at_end=False, # No checkpoint saving means we take the last one
    group_by_length=True,
    dataloader_num_workers=0,
    report_to="none",
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
# ENTRENAR
# ============================================================================

print("\n🚀 INICIANDO FINE-TUNING (PEFT MODE) - 200 STEPS")
trainer.train()

print("\n💾 Guardando modelo final...")
final_model_dir = os.path.join(OUTPUT_DIR, "Llama-3.2-3B-trained-v2")
trainer.model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"✅ Modelo guardado en: {final_model_dir}")

# AUTO-COPY TO CHATBOT APP
CHATBOT_MODEL_DIR = r"d:\FineTuning\chatbot\chatbot_app\Llama-3.2-3B-trained-v2"
print(f"\n📦 Copiando modelo a la aplicación de chatbot: {CHATBOT_MODEL_DIR}")
if os.path.exists(CHATBOT_MODEL_DIR):
    shutil.rmtree(CHATBOT_MODEL_DIR)
shutil.copytree(final_model_dir, CHATBOT_MODEL_DIR)
print("✅ Copia completada exitosamente.")

