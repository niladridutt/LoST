import torch

def configure_compute_backend():
    """Configure PyTorch compute backend settings for CUDA."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        raise ValueError("No CUDA available")

def get_device():
    """Get the device to use for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        raise ValueError("No CUDA available")