# Análisis Técnico del Script de Fine-Tuning: `train_peft_v3.py`

## Propósito General

Este script implementa el proceso de **Fine-Tuning** (ajuste fino) del modelo de lenguaje **Llama-3.2-3B-Instruct** utilizando la técnica **QLoRA** (Quantized Low-Rank Adaptation). El objetivo es especializar un modelo de propósito general en el dominio específico de normativa universitaria, manteniendo un consumo eficiente de recursos computacionales.

---

## 1. Importación de Librerías

```python
import torch
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
```

### Explicación de cada librería:

| Librería | Función |
|----------|---------|
| `torch` | Framework de Deep Learning para operaciones tensoriales y gestión de GPU. |
| `datasets` | Librería de Hugging Face para cargar y manipular conjuntos de datos de entrenamiento. |
| `transformers` | Proporciona acceso a modelos preentrenados (Llama, GPT, BERT) y utilidades de tokenización. |
| `peft` | **Parameter-Efficient Fine-Tuning**: Implementa LoRA y otras técnicas para entrenar solo una fracción de los parámetros. |
| `trl` | **Transformer Reinforcement Learning**: Contiene `SFTTrainer` optimizado para Supervised Fine-Tuning. |

---

## 2. Configuración de Hiperparámetros

```python
# Paths
TRAIN_FILE = r"d:\FineTuning\dataset\train_v3_combined_split.jsonl"
TEST_FILE = r"d:\FineTuning\dataset\test_v3_combined_split.jsonl"
OUTPUT_DIR = r"d:\FineTuning\outputs_peft"

# Model
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 512

# QLoRA Config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_STEPS = 2000
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5
```

### Justificación de cada parámetro:

#### **Modelo Base**
- **`unsloth/Llama-3.2-3B-Instruct`**: Versión optimizada del modelo Llama 3.2 con 3 billones de parámetros. El sufijo "Instruct" indica que fue pre-entrenado para seguir instrucciones.

#### **Longitud de Secuencia**
- **`MAX_SEQ_LENGTH = 512`**: Limita el contexto a 512 tokens. Dado que las respuestas sobre normativa son concisas, este valor es suficiente y reduce el consumo de memoria VRAM.

#### **Parámetros LoRA**

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `LORA_R` | 32 | **Rango de las matrices de adaptación**. Un valor mayor permite aprender patrones más complejos, pero consume más memoria. 32 es un balance entre capacidad y eficiencia. |
| `LORA_ALPHA` | 64 | **Factor de escala**. Controla cuánto influyen los adaptadores LoRA en la salida final. Típicamente se usa `alpha = 2 * r`. |
| `LORA_DROPOUT` | 0.05 | **Regularización**. Desactiva aleatoriamente el 5% de las conexiones durante el entrenamiento para evitar sobreajuste. |
| `TARGET_MODULES` | [q, k, v, o, gate, up, down] | **Capas modificadas**. Se aplican adaptadores a las proyecciones de atención (Query, Key, Value, Output) y a las capas Feed-Forward (gate, up, down). |

#### **Hiperparámetros de Entrenamiento**

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `BATCH_SIZE` | 1 | Limitado por la memoria de GPU (8GB). |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | Simula un batch efectivo de 2 acumulando gradientes antes de actualizar pesos. |
| `LEARNING_RATE` | 1e-4 | Tasa de aprendizaje conservadora para evitar "olvidar" el conocimiento previo del modelo. |
| `MAX_STEPS` | 2000 | Número total de iteraciones. Con un dataset pequeño, esto equivale a múltiples épocas. |
| `WARMUP_STEPS` | 100 | Durante los primeros 100 pasos, el learning rate aumenta gradualmente desde 0 hasta 1e-4. Esto estabiliza el inicio del entrenamiento. |
| `WEIGHT_DECAY` | 0.01 | Penalización L2 sobre los pesos para evitar sobreajuste. |
| `MAX_GRAD_NORM` | 0.5 | Recorte de gradientes (Gradient Clipping). Previene explosiones de gradiente limitando su magnitud máxima. |

---

## 3. Configuración de Cuantización (QLoRA)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### Explicación Técnica:

**QLoRA** combina dos técnicas:
1. **Cuantización a 4 bits**: Reduce la precisión de los pesos del modelo de 32 bits (float32) a 4 bits, disminuyendo el uso de memoria ~8 veces.
2. **LoRA**: Entrena solo matrices adicionales pequeñas en lugar de todos los parámetros.

| Parámetro | Función |
|-----------|---------|
| `load_in_4bit=True` | Carga el modelo con pesos cuantizados a 4 bits. |
| `bnb_4bit_quant_type="nf4"` | **NormalFloat4**: Tipo de cuantización optimizado para distribuciones normales de pesos, típicas en redes neuronales. |
| `bnb_4bit_compute_dtype=torch.float16` | Los cálculos se realizan en precisión FP16, balanceando velocidad y precisión. |
| `bnb_4bit_use_double_quant=True` | Aplica una segunda cuantización a las constantes de cuantización, ahorrando memoria adicional. |

### Impacto en Recursos:
- **Sin cuantización**: ~12 GB VRAM (no cabe en RTX 2070 SUPER).
- **Con QLoRA 4-bit**: ~5-6 GB VRAM (viable en GPU de consumo).

---

## 4. Carga del Modelo Base

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

### Detalles Importantes:

1. **`device_map={"": 0}`**: Asigna todo el modelo a la GPU 0. Para modelos más grandes, se puede distribuir entre múltiples GPUs.

2. **`tokenizer.pad_token = tokenizer.eos_token`**: Llama no define un token de padding por defecto. Se reutiliza el token de fin de secuencia (`<eos>`) para este propósito.

3. **`padding_side="right"`**: El padding se añade al final de las secuencias. Esto es estándar para modelos causales (autoregresivos).

---

## 5. Preparación para Entrenamiento en Baja Precisión

```python
model = prepare_model_for_kbit_training(model)
```

Esta función de PEFT realiza ajustes necesarios para entrenar un modelo cuantizado:
- Congela los pesos originales del modelo.
- Habilita el cálculo de gradientes solo para los adaptadores LoRA.
- Ajusta las capas de normalización para funcionar en precisión mixta.

---

## 6. Configuración de LoRA

```python
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
```

### Salida Esperada:
```
trainable params: 83,886,080 || all params: 3,213,004,800 || trainable%: 2.61%
```

**Interpretación**: Solo el 2.61% de los parámetros (las matrices LoRA) serán actualizados durante el entrenamiento. El 97.39% restante permanece congelado, preservando el conocimiento general del modelo.

---

## 7. Carga del Dataset

```python
train_dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')
eval_dataset = load_dataset('json', data_files=TEST_FILE, split='train')
```

El dataset está en formato JSONL (JSON Lines), donde cada línea contiene un campo `text` con el prompt completo en formato Llama 3:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instrucción del sistema}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{pregunta}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{respuesta}<|eot_id|>
```

---

## 8. Argumentos de Entrenamiento

```python
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    optim="paged_adamw_8bit",
    fp16=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    save_strategy="no",
    group_by_length=True,
)
```

### Parámetros Clave para Tesis:

| Parámetro | Explicación Académica |
|-----------|----------------------|
| `lr_scheduler_type="cosine"` | El learning rate sigue una curva coseno, decayendo suavemente desde el máximo hasta cero. Esto permite exploración inicial y convergencia fina al final. |
| `optim="paged_adamw_8bit"` | Variante de Adam que usa paginación de memoria y precisión de 8 bits para los estados del optimizador, reduciendo uso de VRAM. |
| `fp16=True` | Entrenamiento en precisión mixta (Float16). Acelera el cálculo y reduce memoria sin pérdida significativa de precisión. |
| `group_by_length=True` | Agrupa secuencias de longitud similar en el mismo batch, minimizando padding desperdiciado. |

---

## 9. Inicialización del Entrenador

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    callbacks=[loss_callback],
)
```

**SFTTrainer** (Supervised Fine-Tuning Trainer) es una clase especializada de la librería TRL que:
- Maneja automáticamente la tokenización del campo `text`.
- Aplica el formato de pérdida causal (predecir el siguiente token).
- Integra callbacks para logging y visualización.

---

## 10. Ejecución del Entrenamiento

```python
trainer.train()
```

Durante la ejecución:
1. El modelo procesa batches del dataset de entrenamiento.
2. Calcula la **pérdida de entropía cruzada** entre las predicciones y los tokens reales.
3. Propaga gradientes solo a través de los adaptadores LoRA.
4. Cada `eval_steps` iteraciones, evalúa en el conjunto de prueba.

---

## 11. Guardado del Modelo

```python
trainer.model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
```

**Salida**: Solo se guardan los adaptadores LoRA (~185 MB), no el modelo completo (~6 GB). Durante la inferencia, se carga el modelo base desde Hugging Face y se aplican los adaptadores localmente.

---

## Diagrama del Proceso

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING CON QLoRA                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  Modelo Base │    │ Cuantización │    │  Adaptadores │     │
│   │  Llama 3.2   │ -> │   4-bit NF4  │ -> │     LoRA     │     │
│   │   (3B)       │    │              │    │   (r=32)     │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                 ↓               │
│   ┌──────────────────────────────────────────────────────┐     │
│   │                 ENTRENAMIENTO                         │     │
│   │  • 2000 pasos (max_steps)                            │     │
│   │  • Learning rate: 1e-4 con decay coseno              │     │
│   │  • Batch efectivo: 2 (1 * 2 acumulación)             │     │
│   │  • Evaluación cada 200 pasos                         │     │
│   └──────────────────────────────────────────────────────┘     │
│                                                 ↓               │
│   ┌──────────────┐                                             │
│   │   Salida:    │  → Adaptadores LoRA (~185 MB)               │
│   │   Guardado   │  → Tokenizer                                │
│   └──────────────┘                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Consideraciones para la Tesis

### Ventajas del Enfoque QLoRA:
1. **Eficiencia de memoria**: Permite entrenar modelos de billones de parámetros en hardware de consumo.
2. **Preservación del conocimiento**: Al congelar los pesos originales, se mantiene la capacidad lingüística general.
3. **Rapidez**: El entrenamiento de 2000 pasos toma aproximadamente 2-3 horas en una RTX 2070 SUPER.

### Limitaciones a Mencionar:
1. **Dependencia del modelo base**: La calidad final depende de qué tan bien el modelo base puede generalizar.
2. **Cuantización**: Aunque NF4 minimiza la pérdida de precisión, existe una pequeña degradación teórica.
3. **Dominio cerrado**: El modelo está especializado en normativa UTN y puede fallar en otros dominios.

---

## Referencias Técnicas

- Dettmers, T. et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs". arXiv:2305.14314
- Hu, E. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models". arXiv:2106.09685
- Meta AI (2024). "Llama 3.2 Model Card". Hugging Face Documentation.
