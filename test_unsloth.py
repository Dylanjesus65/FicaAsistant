#!/usr/bin/env python
# Test rápido de Unsloth
import sys
try:
    print("Importando torch...")
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
    
    print("\nImportando Unsloth...")
    from unsloth import FastLanguageModel
    print("✅ Unsloth importado correctamente!")
    
    print("\n🎉 ¡TODO FUNCIONA! Listo para entrenar.")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
