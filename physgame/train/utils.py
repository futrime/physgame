import torch


def flush_right(mask: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Shift non-zero elements in the mask and corresponding tensors to the right.

    Args:
        mask (`torch.Tensor`):
            2D tensor (binary mask) with shape `(N, M)`.
        *tensors (`torch.Tensor`)
            One or more 2D tensors with the same shape as `mask`. These tensors will be processed alongside `mask`,
            with non-zero values shifted and excess zero columns truncated in the same manner.

    Returns:
        `torch.Tensor`:
            Updated binary mask with non-zero values flushed to the right and leading zero columns removed.
        `*torch.Tensor`
            Updated tensors, processed in the same way as the mask.
    """
    # Create copy of mask and tensors
    mask = mask.clone()
    tensors_list = [t.clone() for t in tensors]

    # Shift non-zero values to the right
    for i in range(mask.size(0)):
        if torch.any(mask[i]):
            last_one_idx = int(torch.nonzero(mask[i])[-1].item())
            shift_amount = mask.size(1) - 1 - last_one_idx
            mask[i] = torch.roll(mask[i], shifts=shift_amount)
            for tensor in tensors_list:
                tensor[i] = torch.roll(tensor[i], shifts=shift_amount)

    # Get the first column idx that is not all zeros and remove every column before that
    non_empty_cols = torch.sum(mask, dim=0) > 0
    first_non_empty_col = (
        torch.nonzero(non_empty_cols)[0].item() if non_empty_cols.any() else 0
    )
    mask = mask[:, first_non_empty_col:]
    for i, tensor in enumerate(tensors_list):
        tensors_list[i] = tensor[:, first_non_empty_col:]

    if not tensors:
        return (mask,)
    else:
        return (mask, *tensors_list)

