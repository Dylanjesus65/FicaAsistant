import os
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset

# Configuración de logs
logging.set_verbosity(logging.INFO)

def train():
    # Cargar configuración
    print("Cargando configuración...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Cargar dataset local (JSONL)
    print(f"Cargando dataset local desde: {config['dataset_name']}...")
    try:
        # Para archivos JSONL locales:
        dataset_files = [os.path.join(config['dataset_name'], f) for f in os.listdir(config['dataset_name']) if f.endswith('.jsonl')]
        dataset = load_dataset('json', data_files=dataset_files, split='train')
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    # Configuración de cuantización (4-bit)
    compute_dtype = getattr(torch, config['bnb_4bit_compute_dtype'])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['use_4bit'],
        bnb_4bit_quant_type=config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['use_nested_quant'],
    )

    # Cargar modelo base
    print(f"Cargando modelo base: {config['model_name']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        device_map=config['device_map']
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Cargar tokenizador
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix para fp16

    # Configuración LoRA
    peft_config = LoraConfig(
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        r=config['lora_r'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Argumentos de entrenamiento
    training_arguments = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        optim=config['optim'],
        save_steps=config['save_steps'],
        logging_steps=config['logging_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        fp16=False,
        bf16=False,
        max_grad_norm=config['max_grad_norm'],
        max_steps=-1,
        warmup_ratio=config['warmup_ratio'],
        group_by_length=config['group_by_length'],
        lr_scheduler_type=config['lr_scheduler_type'],
        report_to="tensorboard"
    )

    # Inicializar Trainer (SFT)
    print("Inicializando SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config['max_seq_length'],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config['packing'],
    )

    # Entrenar
    print("Iniciando entrenamiento...")
    trainer.train()

    # Guardar modelo
    print(f"Guardando modelo en {config['new_model_name']}...")
    trainer.model.save_pretrained(config['new_model_name'])
    tokenizer.save_pretrained(config['new_model_name'])
    
    print("¡Entrenamiento completado!")

if __name__ == "__main__":
    train()
