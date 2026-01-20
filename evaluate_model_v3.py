"""
==============================================================================
EVALUACIÓN EXHAUSTIVA DEL MODELO - MLOps Senior Level
==============================================================================
Script modular para evaluación cuantitativa del modelo Llama-3.2-3B-entrenado-v3

Métricas implementadas:
- Perplexity (Ventana Deslizante)
- BLEU Score
- ROUGE Score (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore (Precision, Recall, F1)

Autor: MLOps Pipeline
Fecha: 2026-01-19
==============================================================================
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Transformers & PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Métricas
from evaluate import load as load_metric

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuración centralizada para la evaluación"""
    # Modelo
    base_model_id: str = "unsloth/Llama-3.2-3B-Instruct"
    adapter_path: str = r"d:\FineTuning\chatbot\chatbot_app\Llama-3.2-3B-entrenado-v3"
    
    # Dataset
    test_file: str = r"d:\FineTuning\dataset\test_v3_combined_split.jsonl"
    
    # Perplexity
    stride: int = 256  # Ventana deslizante
    max_length: int = 512
    
    # Generación
    max_new_tokens: int = 150
    temperature: float = 0.3
    top_p: float = 0.9
    
    # Límites
    max_samples: int = 30  # Para evaluación rápida
    
    # System Prompt
    system_prompt: str = "Eres FicaAsistant de la UTN. Responde solo con información de la normativa oficial. Si no sabes, di: 'No tengo esa información en la normativa.'"


# ============================================================================
# MÓDULO 1: CARGADOR DE MODELO
# ============================================================================

class ModelLoader:
    """Clase responsable de cargar el modelo y tokenizer"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load(self) -> Tuple[any, any]:
        """Carga el modelo base con adaptador PEFT"""
        print("\n" + "="*60)
        print("📦 CARGANDO MODELO")
        print("="*60)
        
        # Configuración de cuantización
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"   Base Model: {self.config.base_model_id}")
        print(f"   Adapter: {os.path.basename(self.config.adapter_path)}")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.adapter_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo base
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_id,
            quantization_config=bnb_config,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Aplicar adaptador PEFT
        self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path)
        self.model.eval()
        
        print("   ✅ Modelo cargado correctamente")
        
        return self.model, self.tokenizer


# ============================================================================
# MÓDULO 2: CARGADOR DE DATOS
# ============================================================================

class DataLoader:
    """Clase responsable de cargar y procesar el dataset de prueba"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def load_test_data(self) -> List[Dict[str, str]]:
        """Carga y parsea el dataset de prueba"""
        print(f"\n📂 Cargando dataset: {os.path.basename(self.config.test_file)}")
        
        samples = []
        with open(self.config.test_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                data = json.loads(line)
                qa_pair = self._extract_qa(data['text'])
                if qa_pair:
                    samples.append(qa_pair)
        
        # Limitar muestras
        samples = samples[:self.config.max_samples]
        print(f"   ✅ Muestras cargadas: {len(samples)}")
        
        return samples
    
    def _extract_qa(self, text: str) -> Optional[Dict[str, str]]:
        """Extrae pregunta y respuesta del formato Llama 3"""
        try:
            # Extraer pregunta (user)
            user_start = text.find("<|start_header_id|>user<|end_header_id|>")
            user_end = text.find("<|eot_id|>", user_start)
            question = text[user_start + len("<|start_header_id|>user<|end_header_id|>"):user_end].strip()
            
            # Extraer respuesta (assistant)
            asst_start = text.find("<|start_header_id|>assistant<|end_header_id|>")
            asst_end = text.find("<|eot_id|>", asst_start)
            reference = text[asst_start + len("<|start_header_id|>assistant<|end_header_id|>"):asst_end].strip()
            
            if question and reference:
                return {"question": question, "reference": reference}
        except Exception:
            pass
        return None


# ============================================================================
# MÓDULO 3: GENERADOR DE RESPUESTAS
# ============================================================================

class ResponseGenerator:
    """Clase responsable de generar respuestas del modelo"""
    
    def __init__(self, model, tokenizer, config: EvaluationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def generate(self, question: str) -> str:
        """Genera una respuesta para una pregunta dada"""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": question}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_batch(self, samples: List[Dict]) -> List[Dict]:
        """Genera respuestas para un batch de muestras"""
        print("\n🤖 Generando respuestas del modelo...")
        
        results = []
        for sample in tqdm(samples, desc="Generando"):
            prediction = self.generate(sample["question"])
            results.append({
                "question": sample["question"],
                "reference": sample["reference"],
                "prediction": prediction
            })
        
        return results


# ============================================================================
# MÓDULO 4: CALCULADOR DE PERPLEXITY (VENTANA DESLIZANTE)
# ============================================================================

class PerplexityCalculator:
    """
    Calcula Perplexity usando el método de ventana deslizante.
    Este método es más preciso para textos largos ya que evita
    los efectos de borde al procesar tokens en contexto.
    """
    
    def __init__(self, model, tokenizer, config: EvaluationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
    def calculate_sliding_window(self, texts: List[str]) -> Dict[str, float]:
        """
        Calcula perplexity usando ventana deslizante.
        
        El stride define cuántos tokens nuevos se evalúan en cada paso.
        Un stride menor = más preciso pero más lento.
        """
        print("\n📊 Calculando Perplexity (Ventana Deslizante)...")
        
        nlls = []  # Negative log-likelihoods
        total_tokens = 0
        
        for text in tqdm(texts, desc="PPL"):
            # Tokenizar
            encodings = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length * 2  # Permitir texto más largo
            )
            
            seq_len = encodings.input_ids.size(1)
            
            if seq_len < 2:
                continue
            
            prev_end_loc = 0
            
            for begin_loc in range(0, seq_len, self.config.stride):
                end_loc = min(begin_loc + self.config.max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # Tokens a evaluar
                
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                
                # Enmascarar tokens que ya fueron evaluados
                target_ids[:, :-trg_len] = -100
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len
                
                nlls.append(neg_log_likelihood.item())
                total_tokens += trg_len
                
                prev_end_loc = end_loc
                
                if end_loc >= seq_len:
                    break
        
        # Calcular perplexity promedio
        if total_tokens > 0 and nlls:
            avg_nll = sum(nlls) / total_tokens
            perplexity = float(np.exp(avg_nll))
        else:
            perplexity = float('inf')
        
        # Limitar valores extremos
        perplexity = min(perplexity, 10000.0)
        
        return {
            "perplexity": round(perplexity, 4),
            "avg_nll": round(sum(nlls) / len(nlls) if nlls else 0, 4),
            "total_tokens": total_tokens
        }


# ============================================================================
# MÓDULO 5: CALCULADOR DE MÉTRICAS LÉXICAS (BLEU, ROUGE)
# ============================================================================

class LexicalMetricsCalculator:
    """Calcula métricas de superposición léxica: BLEU y ROUGE"""
    
    def __init__(self):
        print("\n📦 Cargando métricas léxicas...")
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calcula BLEU Score.
        BLEU mide la precisión de n-gramas entre predicción y referencia.
        """
        print("\n📊 Calculando BLEU Score...")
        
        # Tokenizar para BLEU
        pred_tokens = [p.split() for p in predictions]
        ref_tokens = [[r.split()] for r in references]  # BLEU espera lista de listas
        
        try:
            result = self.bleu.compute(predictions=pred_tokens, references=ref_tokens)
            return {
                "bleu": round(result["bleu"] * 100, 2),
                "bleu_1": round(result["precisions"][0] * 100, 2) if result["precisions"] else 0,
                "bleu_2": round(result["precisions"][1] * 100, 2) if len(result["precisions"]) > 1 else 0,
                "bleu_3": round(result["precisions"][2] * 100, 2) if len(result["precisions"]) > 2 else 0,
                "bleu_4": round(result["precisions"][3] * 100, 2) if len(result["precisions"]) > 3 else 0,
            }
        except Exception as e:
            print(f"   ⚠️ Error en BLEU: {e}")
            return {"bleu": 0, "bleu_1": 0, "bleu_2": 0, "bleu_3": 0, "bleu_4": 0}
    
    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calcula ROUGE Scores.
        - ROUGE-1: Superposición de unigramas
        - ROUGE-2: Superposición de bigramas
        - ROUGE-L: Subsecuencia común más larga
        """
        print("📊 Calculando ROUGE Scores...")
        
        try:
            result = self.rouge.compute(predictions=predictions, references=references)
            return {
                "rouge1": round(result["rouge1"] * 100, 2),
                "rouge2": round(result["rouge2"] * 100, 2),
                "rougeL": round(result["rougeL"] * 100, 2),
                "rougeLsum": round(result["rougeLsum"] * 100, 2),
            }
        except Exception as e:
            print(f"   ⚠️ Error en ROUGE: {e}")
            return {"rouge1": 0, "rouge2": 0, "rougeL": 0, "rougeLsum": 0}


# ============================================================================
# MÓDULO 6: CALCULADOR DE BERTSCORE (SIMILITUD SEMÁNTICA)
# ============================================================================

class BERTScoreCalculator:
    """
    Calcula BERTScore para similitud semántica.
    BERTScore usa embeddings de BERT para comparar semánticamente
    las predicciones con las referencias, capturando sinónimos y
    paráfrasis que BLEU/ROUGE no pueden detectar.
    """
    
    def __init__(self):
        print("📦 Cargando BERTScore...")
        self.bertscore = load_metric("bertscore")
        
    def calculate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calcula BERTScore (Precision, Recall, F1)"""
        print("\n📊 Calculando BERTScore (Similitud Semántica)...")
        
        try:
            result = self.bertscore.compute(
                predictions=predictions,
                references=references,
                lang="es",  # Español
                model_type="bert-base-multilingual-cased"
            )
            
            return {
                "bertscore_precision": round(np.mean(result["precision"]) * 100, 2),
                "bertscore_recall": round(np.mean(result["recall"]) * 100, 2),
                "bertscore_f1": round(np.mean(result["f1"]) * 100, 2),
            }
        except Exception as e:
            print(f"   ⚠️ Error en BERTScore: {e}")
            return {"bertscore_precision": 0, "bertscore_recall": 0, "bertscore_f1": 0}


# ============================================================================
# MÓDULO 7: GENERADOR DE REPORTE
# ============================================================================

class ReportGenerator:
    """Genera el reporte consolidado de métricas"""
    
    @staticmethod
    def generate(metrics: Dict[str, any], samples: int) -> None:
        """Imprime el reporte consolidado en consola"""
        
        print("\n" + "="*70)
        print("📊 REPORTE DE EVALUACIÓN - Llama-3.2-3B-entrenado-v3")
        print("="*70)
        
        print(f"\n📋 Resumen General:")
        print(f"   Muestras evaluadas: {samples}")
        
        # Perplexity
        print(f"\n🎯 PERPLEXITY (Ventana Deslizante)")
        print(f"   └─ Perplexity:    {metrics.get('perplexity', 'N/A')}")
        print(f"   └─ Avg NLL:       {metrics.get('avg_nll', 'N/A')}")
        print(f"   └─ Total Tokens:  {metrics.get('total_tokens', 'N/A')}")
        
        # BLEU
        print(f"\n📝 BLEU SCORE (Superposición de N-gramas)")
        print(f"   └─ BLEU:          {metrics.get('bleu', 'N/A')}%")
        print(f"   └─ BLEU-1:        {metrics.get('bleu_1', 'N/A')}%")
        print(f"   └─ BLEU-2:        {metrics.get('bleu_2', 'N/A')}%")
        print(f"   └─ BLEU-3:        {metrics.get('bleu_3', 'N/A')}%")
        print(f"   └─ BLEU-4:        {metrics.get('bleu_4', 'N/A')}%")
        
        # ROUGE
        print(f"\n📝 ROUGE SCORE (Superposición Léxica)")
        print(f"   └─ ROUGE-1:       {metrics.get('rouge1', 'N/A')}%")
        print(f"   └─ ROUGE-2:       {metrics.get('rouge2', 'N/A')}%")
        print(f"   └─ ROUGE-L:       {metrics.get('rougeL', 'N/A')}%")
        print(f"   └─ ROUGE-Lsum:    {metrics.get('rougeLsum', 'N/A')}%")
        
        # BERTScore
        print(f"\n🧠 BERTSCORE (Similitud Semántica)")
        print(f"   └─ Precision:     {metrics.get('bertscore_precision', 'N/A')}%")
        print(f"   └─ Recall:        {metrics.get('bertscore_recall', 'N/A')}%")
        print(f"   └─ F1:            {metrics.get('bertscore_f1', 'N/A')}%")
        
        # Interpretación
        print("\n" + "="*70)
        print("📈 INTERPRETACIÓN DE RESULTADOS")
        print("="*70)
        
        ppl = metrics.get('perplexity', float('inf'))
        if ppl < 5:
            ppl_eval = "🟢 Excelente (modelo muy seguro)"
        elif ppl < 20:
            ppl_eval = "🟡 Bueno (modelo confiable)"
        elif ppl < 100:
            ppl_eval = "🟠 Aceptable (puede mejorar)"
        else:
            ppl_eval = "🔴 Alto (modelo incierto)"
        
        bert_f1 = metrics.get('bertscore_f1', 0)
        if bert_f1 > 80:
            bert_eval = "🟢 Excelente similitud semántica"
        elif bert_f1 > 60:
            bert_eval = "🟡 Buena similitud semántica"
        elif bert_f1 > 40:
            bert_eval = "🟠 Similitud moderada"
        else:
            bert_eval = "🔴 Baja similitud semántica"
        
        print(f"   Perplexity: {ppl_eval}")
        print(f"   BERTScore:  {bert_eval}")
        
        print("\n" + "="*70)
        print("✅ EVALUACIÓN COMPLETADA")
        print("="*70)


# ============================================================================
# ORQUESTADOR PRINCIPAL
# ============================================================================

class ModelEvaluator:
    """Orquestador principal que coordina todos los módulos de evaluación"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        
    def run(self) -> Dict[str, any]:
        """Ejecuta la evaluación completa"""
        
        print("\n" + "="*70)
        print("🚀 EVALUACIÓN EXHAUSTIVA DEL MODELO")
        print("   Modelo: Llama-3.2-3B-entrenado-v3")
        print("="*70)
        
        # 1. Cargar modelo
        loader = ModelLoader(self.config)
        model, tokenizer = loader.load()
        
        # 2. Cargar datos
        data_loader = DataLoader(self.config)
        samples = data_loader.load_test_data()
        
        if not samples:
            print("❌ No hay muestras para evaluar")
            return {}
        
        # 3. Generar respuestas
        generator = ResponseGenerator(model, tokenizer, self.config)
        results = generator.generate_batch(samples)
        
        predictions = [r["prediction"] for r in results]
        references = [r["reference"] for r in results]
        
        # 4. Calcular Perplexity (ventana deslizante)
        ppl_calc = PerplexityCalculator(model, tokenizer, self.config)
        ppl_texts = [r["prediction"] for r in results]
        ppl_metrics = ppl_calc.calculate_sliding_window(ppl_texts)
        
        # 5. Calcular BLEU y ROUGE
        lexical_calc = LexicalMetricsCalculator()
        bleu_metrics = lexical_calc.calculate_bleu(predictions, references)
        rouge_metrics = lexical_calc.calculate_rouge(predictions, references)
        
        # 6. Calcular BERTScore
        bert_calc = BERTScoreCalculator()
        bert_metrics = bert_calc.calculate(predictions, references)
        
        # 7. Consolidar métricas
        all_metrics = {
            **ppl_metrics,
            **bleu_metrics,
            **rouge_metrics,
            **bert_metrics
        }
        
        # 8. Generar reporte
        ReportGenerator.generate(all_metrics, len(samples))
        
        # 9. Guardar métricas
        output_file = os.path.join(
            os.path.dirname(self.config.adapter_path),
            "metrics_report_v3.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Métricas guardadas en: {output_file}")
        
        return all_metrics


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Configurar evaluación
    config = EvaluationConfig(
        max_samples=25,  # Ajustar según necesidad
        stride=128,      # Ventana deslizante más precisa
        max_length=512,
        max_new_tokens=150,
        temperature=0.3,
    )
    
    # Ejecutar evaluación
    evaluator = ModelEvaluator(config)
    metrics = evaluator.run()
