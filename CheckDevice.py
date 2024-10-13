import torch

def check_devices():
    print("Available devices:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print("No GPU available.")

check_devices()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model is running on device: {device}")

import torch

print("PyTorch CUDA version:", torch.version.cuda)
print("Is cuDNN enabled:", torch.backends.cudnn.enabled)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")