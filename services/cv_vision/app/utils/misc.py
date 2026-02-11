import torch
import torch.nn.functional as F


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    Wrapper around torch.nn.functional.interpolate for resizing tensors.
    
    Args:
        input: Input tensor
        size: Target size (H, W)
        scale_factor: Scale factor for resizing
        mode: Interpolation mode ('nearest', 'bilinear', 'bicubic', 'trilinear')
        align_corners: Whether to align corners (for 'bilinear' and 'bicubic')
    
    Returns:
        Resized tensor
    """
    if mode in ['bilinear', 'bicubic', 'trilinear']:
        return F.interpolate(input, size=size, scale_factor=scale_factor, 
                           mode=mode, align_corners=align_corners)
    else:
        return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode)
