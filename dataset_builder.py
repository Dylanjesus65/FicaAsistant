"""
Dataset Builder V5 - Sistema Optimizado para Fine-Tuning
Usa System Prompt CORTO para maximizar aprendizaje del contenido real.
"""

import os
import json
import re
import random
import ftfy
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

# Configuración
DATASET_DIR = r"d:\FineTuning\Dataset"
OUTPUT_FILE = r"d:\FineTuning\dataset\train.jsonl"
POPPLER_BIN_PATH = r"d:\FineTuning\poppler\poppler-24.08.0\Library\bin"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# SYSTEM PROMPT CORTO Y EFECTIVO (para entrenamiento)
# El modelo aprenderá a responder así, y en inferencia se puede usar un prompt más largo
SYSTEM_PROMPT = "Eres FicaAsistant, asistente de normativa de la UTN. Responde con precisión según los reglamentos oficiales."

# Palabras clave de inicio estricto
START_KEYWORDS = ["RESUELVE", "ACUERDA", "TÍTULO I", "TITULO I", "CAPÍTULO I", "CAPITULO I"]
# Regex Artículos
ARTICLE_PATTERN = r'(?:^|\n)(ART[ÍI]CULO\s+\d+|DISPOSICI[ÓO]N\s+\w+(?:\s+\w+)?)[.\s-]*'

def fix_encoding(text):
    return ftfy.fix_text(text)

def clean_filename(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'^\d+\.?', '', name)
    name = name.replace('_', ' ').replace('-', ' ')
    return name.strip().title()

def extract_text_from_pdf(pdf_path):
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
    except Exception as e:
        print(f"Error leyendo PDF {pdf_path}: {e}")
        return ""

    if len(text_content) < 100: 
        try:
            if os.path.exists(POPPLER_BIN_PATH):
                images = convert_from_path(pdf_path, poppler_path=POPPLER_BIN_PATH)
            else:
                images = convert_from_path(pdf_path)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img, lang='spa') + "\n"
            return ocr_text
        except Exception:
            return "" 
    return text_content

def clean_noise(text):
    text = fix_encoding(text)
    lines = text.split('\n')
    cleaned_lines = []
    
    skip_phrases = [
        "UNIVERSIDAD TÉCNICA DEL NORTE", "RESOLUCIÓN NRO", "RECTOR", "SECRETARIO",
        "PÁGINA", "PAGE", "DIRECCIÓN", "IBARRA", "ACREDITADA"
    ]
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean: continue
        if re.match(r'^\d+$', line_clean): continue
        
        is_noise = False
        for phrase in skip_phrases:
            if phrase in line_clean.upper():
                is_noise = True
                break
        
        if line_clean.endswith('-') and not is_noise:
             cleaned_lines.append(line_clean[:-1] + "##HYPHEN##")
             continue

        if not is_noise:
            cleaned_lines.append(line_clean)
            
    full_text = "\n".join(cleaned_lines)
    full_text = full_text.replace("##HYPHEN##\n", "").replace("##HYPHEN##", "")
    return full_text

def filter_preamble(text):
    text_upper = text.upper()
    best_idx = -1
    for kw in START_KEYWORDS:
        idx = text_upper.find(kw)
        if idx != -1:
            if best_idx == -1 or idx < best_idx:
                best_idx = idx
    if best_idx != -1: return text[best_idx:]
    return text

def extract_topic(article_title, content):
    first_line = content.split('.')[0] if '.' in content else content[:80]
    topic = re.sub(r'[^\w\s]', ' ', first_line).strip()
    if len(topic) > 60:
        topic = topic[:60] + "..."
    return topic if topic else "este tema"

def create_training_example(question, answer):
    """Crea un ejemplo de entrenamiento en formato Llama 3"""
    entry = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
    )
    return {"text": entry}

def main():
    print("🚀 Iniciando Generación de Dataset V5...")
    dataset_data = []
    pdf_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith('.pdf')]
    
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(DATASET_DIR, pdf_file)
        doc_name = clean_filename(pdf_file)
        
        text = extract_text_from_pdf(pdf_path)
        if not text: continue
        
        filtered_text = filter_preamble(text)
        clean_text = clean_noise(filtered_text)
        
        parts = re.split(f'({ARTICLE_PATTERN})', clean_text, flags=re.IGNORECASE)
        
        i = 1
        while i < len(parts):
            article_title = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            content = re.sub(r'\s+', ' ', content)
            
            if len(content) > 50:
                topic = extract_topic(article_title, content)
                
                # Tipo 1: Pregunta directa sobre artículo
                q1 = f"¿Qué establece el {article_title} del {doc_name}?"
                a1 = f"El {article_title} establece: {content}"
                dataset_data.append(create_training_example(q1, a1))
                
                # Tipo 2: Pregunta por tema
                q2 = f"¿Qué dice la normativa sobre {topic}?"
                a2 = f"Según el {article_title} del {doc_name}: {content}"
                dataset_data.append(create_training_example(q2, a2))
                
                # Tipo 3: Pregunta coloquial
                q3 = random.choice([
                    f"Explícame {topic}",
                    f"¿Cómo funciona lo de {topic}?",
                    f"Necesito saber sobre {topic}"
                ])
                a3 = f"Te explico. El {article_title} indica que: {content}"
                dataset_data.append(create_training_example(q3, a3))
            
            i += 2

    print(f"✅ Dataset Generado: {len(dataset_data)} ejemplos de entrenamiento.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in dataset_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"📁 Guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
