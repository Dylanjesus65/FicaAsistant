"""
Script de Chat Interactivo con Modelo Fine-Tuned (PEFT)
Asistente Virtual UTN-FICA
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys

# Configuración
MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
PEFT_MODEL_ID = r"d:\FineTuning\outputs_peft\final_model" # Ajustar si es necesario

def main():
    print(f"🚀 Iniciando Chatbot UTN-FICA...")
    
    # Check si existe el modelo
    import os
    if not os.path.exists(PEFT_MODEL_ID):
        print(f"⚠️ Error: No se encuentra el modelo en {PEFT_MODEL_ID}")
        print("Asegúrate de que el entrenamiento haya terminado.")
        return

    # 1. Cargar Tokenizer
    print("📚 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Cargar Modelo Base
    print("📦 Cargando modelo base (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    # 3. Cargar Adaptador
    print("🔗 Conectando adaptador LoRA...")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    model.eval()
    
    print("\n✅ ¡Sistema listo! Escribe 'salir' para terminar.\n")
    
    # Loop de chat
    history = []
    
    while True:
        try:
            user_input = input("👤 Alumno: ")
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break
                
            # Formato Llama 3 Chat
            # Simplificado: solo user/assistant actual
            # Para memoria completa, habría que acumular history correctamente
            
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True 
                )
            
            # Decodificar solo la respuesta nueva
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            print(f"🤖 Asistente: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
