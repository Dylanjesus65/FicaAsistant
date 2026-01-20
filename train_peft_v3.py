"""
Fine-Tuning V3 - 2000 Pasos con QLoRA
Modelo: Llama-3.2-3B-entrenado-v3
Optimizado para RTX 2070 SUPER (8GB VRAM)
"""

import os
import shutil
import torch
torch.cuda.empty_cache()
import numpy as np
import matplotlib.pyplot as plt
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
    prepare_model_for_kbit_training
)
from trl import SFTTrainer

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Paths
TRAIN_FILE = r"d:\FineTuning\dataset\train_v3_combined_split.jsonl"
TEST_FILE = r"d:\FineTuning\dataset\test_v3_combined_split.jsonl"
OUTPUT_DIR = r"d:\FineTuning\outputs_peft"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints_v3")

# Model
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 512  # Reducido para respuestas concisas

# QLoRA Config
LORA_R = 32  # Aumentado para más capacidad
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters - 2000 PASOS
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
LEARNING_RATE = 1e-4  # Más bajo para estabilidad
MAX_STEPS = 2000
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5  # Más bajo para estabilidad

# Logging
LOGGING_STEPS = 50
EVAL_STEPS = 200
SAVE_STRATEGY = "no"

# Crear directorios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# CARGAR MODELO
# ============================================================================

print("🚀 Cargando modelo V3...")
print(f"   Modelo: {MODEL_NAME}")
print(f"   Pasos: {MAX_STEPS}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)

print(f"\n⚙️ Configurando LoRA (r={LORA_R}, alpha={LORA_ALPHA}):")
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

print(f"\n📂 Cargando datasets V3:")
train_dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')
eval_dataset = load_dataset('json', data_files=TEST_FILE, split='train')

print(f"✅ Train: {len(train_dataset)} samples")
print(f"✅ Test:  {len(eval_dataset)} samples")

# ============================================================================
# CALLBACKS
# ============================================================================

class LossPlotCallback(TrainerCallback):
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
                perplexity = np.exp(logs["eval_loss"]) if logs["eval_loss"] < 10 else float('inf')
                self.perplexities.append(perplexity)
                print(f"\n📊 Eval (Step {state.global_step}): Loss={logs['eval_loss']:.4f}, PPL={perplexity:.2f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        print("\n📈 Generando gráficos...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        if self.train_steps:
            axes[0].plot(self.train_steps, self.train_losses, 'b-')
            axes[0].set_title("Training Loss")
            axes[0].grid(True)
        
        if self.eval_steps:
            axes[1].plot(self.eval_steps, self.eval_losses, 'orange')
            axes[1].set_title("Eval Loss")
            axes[1].grid(True)
            
            axes[2].plot(self.eval_steps, self.perplexities, 'g-')
            axes[2].set_title("Perplexity")
            axes[2].grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves_v3.png"))
        plt.close()

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    optim="paged_adamw_8bit",
    fp16=True,
    bf16=False,
    logging_dir=os.path.join(OUTPUT_DIR, "logs_v3"),
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy=SAVE_STRATEGY,
    load_best_model_at_end=False,
    group_by_length=True,
    dataloader_num_workers=0,
    report_to="none",
)

# ============================================================================
# TRAINER
# ============================================================================

def formatting_prompts_func(examples):
    return {"text": examples["text"]}

print("\n🎯 Inicializando Trainer V3...")
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

print(f"\n🚀 INICIANDO FINE-TUNING V3 ({MAX_STEPS} PASOS)")
trainer.train()

print("\n💾 Guardando modelo...")
final_model_dir = os.path.join(OUTPUT_DIR, "Llama-3.2-3B-entrenado-v3")
trainer.model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"✅ Modelo guardado: {final_model_dir}")

# Copiar a chatbot
CHATBOT_MODEL_DIR = r"d:\FineTuning\chatbot\chatbot_app\Llama-3.2-3B-entrenado-v3"
print(f"\n📦 Copiando a chatbot...")
if os.path.exists(CHATBOT_MODEL_DIR):
    shutil.rmtree(CHATBOT_MODEL_DIR)
shutil.copytree(final_model_dir, CHATBOT_MODEL_DIR)
print("✅ Copia completada.")
