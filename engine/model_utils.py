import torch


def best_device() -> str:
    """Pick the fastest available device: Apple Silicon GPU > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
