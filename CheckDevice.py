import torch

def check_devices():
    """
    Check and print information about available devices.
    """
    print("Available devices:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i):,} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i):,} bytes")
    else:
        print("No GPU available.")

def main():
    """
    Main function to execute the device check and print relevant CUDA information.
    """
    # Check and display available devices
    check_devices()

    # Determine the appropriate device for model execution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is running on device: {device}")

    # Print additional CUDA-related information
    print("\nPyTorch CUDA version:", torch.version.cuda)
    print("Is cuDNN enabled:", torch.backends.cudnn.enabled)

if __name__ == "__main__":
    main()