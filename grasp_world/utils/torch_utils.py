import torch

def resolve_device(device: torch.device | str | None) -> torch.device:
    """Pick a torch.device, defaulting to CUDA when available."""
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


