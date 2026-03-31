import torch


def load_pretrained_unet(scale: float = 0.5, device=None):
    """Load the pretrained Carvana UNet model.

    Args:
        scale: Must be one of {0.5, 1.0}.
        device: Optional torch.device. If None, `get_device()` is used.

    Returns:
        model, device
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if scale not in {0.5, 1.0}:
        raise ValueError(f"scale must be one of {0.5, 1.0}, got {scale}")

    model = torch.hub.load("milesial/Pytorch-UNet", "unet_carvana", pretrained=True, scale=scale)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model
