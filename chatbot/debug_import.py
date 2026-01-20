
import sys
print(f"Python executable: {sys.executable}")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print(f"Transformers file: {transformers.__file__}")
    from transformers import LlamaForCausalLM
    print("SUCCESS: Imported LlamaForCausalLM")
    import dj_database_url
    print(f"SUCCESS: Imported dj_database_url version {dj_database_url.__version__ if hasattr(dj_database_url, '__version__') else 'unknown'}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
