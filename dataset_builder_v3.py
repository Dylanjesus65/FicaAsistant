"""
Dataset Builder V3 - Optimizado para Anti-Alucinación
- Respuestas CONCISAS y directas
- Corrección ortográfica automática
- Lenguaje simple y claro
- Incluye ejemplos de "no sé" para evitar alucinaciones
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
OUTPUT_FILE = r"d:\FineTuning\dataset\train_v3.jsonl"
POPPLER_BIN_PATH = r"d:\FineTuning\poppler\poppler-24.08.0\Library\bin"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# SYSTEM PROMPT para entrenamiento (conciso)
SYSTEM_PROMPT = "Eres FicaAsistant de la UTN. Responde solo con información de la normativa oficial. Si no sabes, di: 'No tengo esa información en la normativa.'"

# Palabras clave de inicio
START_KEYWORDS = ["RESUELVE", "ACUERDA", "TÍTULO I", "TITULO I", "CAPÍTULO I", "CAPITULO I"]
ARTICLE_PATTERN = r'(?:^|\n)(ART[ÍI]CULO\s+\d+|DISPOSICI[ÓO]N\s+\w+(?:\s+\w+)?)[.\s-]*'

# Diccionario de corrección ortográfica común
SPELLING_CORRECTIONS = {
    "informacion": "información",
    "academico": "académico",
    "academica": "académica",
    "caracter": "carácter",
    "titulo": "título",
    "titulacion": "titulación",
    "matricula": "matrícula",
    "credito": "crédito",
    "creditos": "créditos",
    "periodo": "período",
    "regimen": "régimen",
    "articulo": "artículo",
    "numero": "número",
    "unico": "único",
    "unica": "única",
    "sera": "será",
    "tendra": "tendrá",
    "podra": "podrá",
    "debera": "deberá",
    "asi": "así",
    "tambien": "también",
    "mas": "más",
    "segun": "según",
    "traves": "través",
    "dias": "días",
    "curricula": "curricular",
    "tecnicos": "técnicos",
    "tecnico": "técnico",
    "practica": "práctica",
    "practicas": "prácticas",
    "evaluacion": "evaluación",
    "calificacion": "calificación",
    "aprobacion": "aprobación",
    "resolucion": "resolución",
    "disposicion": "disposición",
    "tramite": "trámite",
    "tramites": "trámites",
    "pagina": "página",
    "minimo": "mínimo",
    "maximo": "máximo",
}

def fix_encoding(text):
    return ftfy.fix_text(text)

def correct_spelling(text):
    """Corrige errores ortográficos comunes"""
    text = fix_encoding(text)
    for wrong, correct in SPELLING_CORRECTIONS.items():
        # Case insensitive replacement preserving case
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(correct, text)
    return text

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
             cleaned_lines.append(line_clean[:-1])
             continue

        if not is_noise:
            cleaned_lines.append(line_clean)
            
    full_text = " ".join(cleaned_lines)
    # Limpiar espacios múltiples y caracteres extraños
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = re.sub(r'[\\|/]{2,}', '', full_text)  # Eliminar // \\ etc
    full_text = re.sub(r'\s*-\s*-\s*', ' - ', full_text)  # Limpiar guiones dobles
    return full_text.strip()

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

def extract_topic(content):
    """Extrae el tema principal (primera frase antes del punto)"""
    first_sentence = content.split('.')[0] if '.' in content else content[:60]
    topic = re.sub(r'[^\w\sáéíóúñÁÉÍÓÚÑ]', '', first_sentence).strip()
    if len(topic) > 50:
        topic = topic[:50] + "..."
    return topic if topic else "normativa"

def summarize_content(content, max_chars=300):
    """Resume el contenido a un máximo de caracteres"""
    content = correct_spelling(content)
    if len(content) <= max_chars:
        return content
    # Cortar en el último punto antes del límite
    truncated = content[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > 100:
        return truncated[:last_period+1]
    return truncated + "..."

def create_training_example(question, answer):
    """Crea ejemplo de entrenamiento en formato Llama 3"""
    question = correct_spelling(question)
    answer = correct_spelling(answer)
    
    entry = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
    )
    return {"text": entry}

def main():
    print("🚀 Generando Dataset V3 (Optimizado Anti-Alucinación)...")
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
            article_title = correct_spelling(parts[i].strip())
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            content = re.sub(r'\s+', ' ', content)
            
            if len(content) > 50:
                topic = extract_topic(content)
                short_content = summarize_content(content, max_chars=250)
                
                # Tipo 1: Pregunta directa (respuesta concisa)
                q1 = f"¿Qué dice el {article_title}?"
                a1 = f"{short_content}"
                dataset_data.append(create_training_example(q1, a1))
                
                # Tipo 2: Pregunta por tema
                q2 = f"¿Qué establece la normativa sobre {topic}?"
                a2 = f"El {article_title} indica: {short_content}"
                dataset_data.append(create_training_example(q2, a2))
                
                # Tipo 3: Pregunta coloquial
                q3 = random.choice([
                    f"Explícame sobre {topic}",
                    f"¿Cómo funciona {topic}?",
                    f"Necesito información de {topic}"
                ])
                a3 = f"Según la normativa: {short_content}"
                dataset_data.append(create_training_example(q3, a3))
            
            i += 2
    
    # AÑADIR EJEMPLOS DE "NO SÉ" PARA EVITAR ALUCINACIONES
    no_se_examples = [
        ("¿Cuál es el horario de atención de la biblioteca?", 
         "No tengo esa información en la normativa. Te recomiendo consultar directamente con la biblioteca."),
        ("¿Cuánto cuesta la matrícula?",
         "No tengo información sobre costos en la normativa que manejo."),
        ("¿Quién es el rector actual?",
         "No tengo esa información. Consulta la página oficial de la UTN."),
        ("¿Dónde queda el edificio de ingeniería?",
         "No cuento con información de ubicaciones en la normativa."),
        ("¿Cuándo son las vacaciones?",
         "No tengo el calendario académico. Consulta con secretaría."),
        ("¿Puedes hacer mi tarea?",
         "Solo puedo responder preguntas sobre la normativa universitaria."),
        ("¿Cuál es la mejor carrera?",
         "No puedo dar opiniones. Solo respondo sobre normativa oficial."),
        ("Háblame de fútbol",
         "Solo puedo ayudarte con temas de normativa universitaria."),
    ]
    
    for q, a in no_se_examples:
        dataset_data.append(create_training_example(q, a))
    
    # AÑADIR EJEMPLOS DE IDENTIDAD
    identity_examples = [
        ("Hola", "¡Hola! Soy FicaAsistant. ¿En qué puedo ayudarte con la normativa?"),
        ("¿Quién eres?", "Soy FicaAsistant, asistente de normativa de la UTN."),
        ("¿Qué puedes hacer?", "Respondo preguntas sobre reglamentos y normativas de la UTN."),
        ("Gracias", "¡De nada! ¿Tienes otra consulta sobre la normativa?"),
        ("Adiós", "¡Hasta luego! Recuerda que estoy aquí para ayudarte con la normativa."),
    ]
    
    for q, a in identity_examples:
        dataset_data.append(create_training_example(q, a))

    print(f"✅ Dataset V3 generado: {len(dataset_data)} ejemplos")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in dataset_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"📁 Guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
