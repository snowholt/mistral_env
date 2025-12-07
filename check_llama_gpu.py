import llama_cpp
import os

print(f"llama_cpp version: {llama_cpp.__version__}")
try:
    # Try to load a dummy model or just check internal flags if possible
    # But easier is to check if we can load a model with n_gpu_layers > 0
    # Or check if the library was built with CUBLAS/CUDA
    pass
except Exception as e:
    print(e)

# Check environment variables
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Check if we can import the low level bindings and check for gpu support
try:
    from llama_cpp import llama_backend_init, Llama
    llama_backend_init(numa=False)
    print("Backend initialized")
    
    model_path = "/home/lumi/.cache/huggingface/hub/models--unsloth--Qwen3-14B-GGUF/snapshots/a04a82c4739b3ef5fa6da7d10261db2c67dd1985/Qwen3-14B-Q4_K_S.gguf"
    if os.path.exists(model_path):
        print(f"Found model at {model_path}")
        try:
            # Try to load with GPU layers
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=1, # Try 1 layer on GPU
                verbose=True
            )
            print("Successfully loaded model with n_gpu_layers=1")
            del llm
        except Exception as e:
            print(f"Failed to load model with GPU: {e}")
    else:
        print("Model file not found")
except Exception as e:
    print(f"Backend init failed: {e}")

