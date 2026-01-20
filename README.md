# 🎓 FicaAsistant - Chatbot de Normativa Universitaria UTN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.40+-orange" alt="Transformers">
  <img src="https://img.shields.io/badge/Django-4.2-green" alt="Django">
  <img src="https://img.shields.io/badge/Model-Llama--3.2--3B-purple" alt="Model">
</p>

Chatbot especializado en normativa universitaria de la Universidad Técnica del Norte, basado en **Llama-3.2-3B-Instruct** fine-tuneado con QLoRA.

> ⚠️ **IMPORTANTE: REQUISITO DE GPU**
> 
> Este proyecto **requiere obligatoriamente** una tarjeta gráfica NVIDIA con soporte CUDA para funcionar correctamente.
> - **Mínimo**: GPU NVIDIA con 8GB VRAM (RTX 2070 Super o superior)
> - **CUDA**: Versión 12.1 o superior
> - **Sin GPU compatible, el proyecto NO funcionará**
> 
> Si no cuentas con una GPU compatible, considera usar servicios en la nube como Google Colab Pro, AWS, o Azure con instancias GPU.

---

## 📋 Tabla de Contenidos

- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Fine-Tuning del Modelo](#-fine-tuning-del-modelo)
- [Ejecutar el Chatbot](#-ejecutar-el-chatbot)
- [Evaluación de Métricas](#-evaluación-de-métricas)
- [Historial de Versiones](#-historial-de-versiones)

---

## 🔧 Requisitos

### Hardware
- GPU NVIDIA con mínimo 8GB VRAM (RTX 2070 Super o superior)
- 16GB RAM mínimo

### Software
- Python 3.12
- CUDA 12.1+
- Windows 10/11 o Linux

---

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/FicaAsistant.git
cd FicaAsistant
```

### 2. Crear entorno virtual para entrenamiento
```bash
python -m venv venv_unsloth
venv_unsloth\Scripts\activate  # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl bitsandbytes datasets evaluate accelerate
pip install bert-score rouge-score sacrebleu
```

### 3. Crear entorno virtual para el chatbot
```bash
python -m venv venv312
venv312\Scripts\activate  # Windows

pip install django channels daphne
pip install torch transformers peft bitsandbytes
```

---

## 🎯 Fine-Tuning del Modelo

### Paso 1: Generar el Dataset
```bash
venv312\Scripts\activate
python dataset_builder_v3.py
python split_dataset_v3.py
```

### Paso 2: Ejecutar Entrenamiento
```bash
venv_unsloth\Scripts\activate
python train_peft_v3.py
```

### Parámetros de Entrenamiento
| Parámetro | Valor |
|-----------|-------|
| Modelo base | unsloth/Llama-3.2-3B-Instruct |
| LoRA r | 32 |
| LoRA alpha | 64 |
| Learning rate | 1e-4 |
| Max steps | 2000 |
| Cuantización | 4-bit (QLoRA) |

---

## 💬 Ejecutar el Chatbot

### Paso 0: Descargar modelos entrenados

> ⚠️ **Los modelos entrenados NO están incluidos en el repositorio** debido a su tamaño.

Descarga el modelo v3 desde Hugging Face:
```bash
# Instalar huggingface_hub si no lo tienes
pip install huggingface_hub

# Descargar el modelo (crea la carpeta automáticamente)
huggingface-cli download Dylanjesus65/FicaAsistant-Llama-3.2-3B-v3 --local-dir chatbot/chatbot_app/Llama-3.2-3B-entrenado-v3
```

O si prefieres entrenar tu propio modelo, sigue las instrucciones en la sección [Fine-Tuning del Modelo](#-fine-tuning-del-modelo).

### Paso 1: Iniciar servidor
```bash
venv312\Scripts\activate
cd chatbot
python -m daphne -b 0.0.0.0 -p 8000 chatbot.asgi:application
```

### Paso 2: Acceder
Abrir navegador en: **http://localhost:8000**

---

## 📊 Evaluación de Métricas

```bash
venv_unsloth\Scripts\activate
python evaluate_model_v3.py
python generar_graficos_metricas.py
```

Ver [METRICAS.md](METRICAS.md) para análisis detallado.

---

## 📈 Historial de Versiones

| Versión | Fecha | Cambios | BERTScore F1 |
|---------|-------|---------|--------------|
| **v3** ⭐ | 2026-01-19 | Dataset optimizado, 2000 pasos, anti-alucinación | **75.98%** |
| v2 | 2026-01-19 | System prompt unificado, 600 pasos | 67.78% |
| v1 | 2026-01-18 | Primera versión, 200 pasos | ~50% |

### ¿Por qué V3 es mejor?
- Dataset optimizado con respuestas concisas
- Sistema anti-alucinación incluido
- Corrección ortográfica automática
- 2000 pasos de entrenamiento
- LoRA r=32 para mayor capacidad

---

## 📄 Licencia

MIT License
