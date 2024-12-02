import torch


def pad_frames(x, divisible_by):
    """
    Pads the tensor `x` along the frame dimension (1) until the number of frames is divisible by `divisible_by`.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width).
        divisible_by (int): The divisor to make the number of frames divisible by.
    
    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, adjusted_num_frames, channels, height, width).
    """
    num_frames = x.shape[1]
    frame_padding = (divisible_by - (num_frames % divisible_by)) % divisible_by
    
    if frame_padding > 0:
        padding_shape = (x.shape[0], frame_padding, x.shape[2], x.shape[3], x.shape[4])
        x_padding = torch.zeros(padding_shape, device=x.device)  # Ensure padding is on the same device
        x = torch.cat((x, x_padding), dim=1)
    
    assert x.shape[1] % divisible_by == 0, (
        f'Frame number mismatch: got {x.shape[1]} frames, not divisible by {divisible_by}.'
    )
    return x