import torch


def segment_coo(src, index, out=None, dim_size=None, reduce="sum", dim=0):
    """
    Segment operation that aggregates values from `src` tensor based on `index` tensor.

    Parameters:
    - src (torch.Tensor): Input tensor whose elements are to be aggregated.
    - index (torch.Tensor): Tensor containing indices to aggregate data along.
    - out (Optional[torch.Tensor]): Output tensor to store the result.
    - dim_size (Optional[int]): The size of the dimension for the output tensor.
    - reduce (str): Reduction operation ('sum', 'mean', 'min', 'max').
    - dim (int): Dimension along which to perform the scatter operation.

    Returns:
    - torch.Tensor: Result of the aggregation.
    """
    if dim_size is None:
        dim_size = index.max().item() + 1

    # Ensuring `out` tensor is initialized correctly:
    if out is None:
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)

    # Expanding index to match src dimensions:
    expanded_index = index.unsqueeze(-1).expand_as(src)

    if reduce == "sum":
        out.scatter_add_(dim, expanded_index, src)
    elif reduce == "mean":
        counts = (
            torch.bincount(index, minlength=dim_size)
            .float()
            .unsqueeze(-1)
            .expand_as(out)
        )
        out.scatter_add_(dim, expanded_index, src)
        out = out / counts.clamp(min=1)  # Avoid division by zero
    elif reduce == "min":
        initial_val = float("inf")
        out.fill_(initial_val)
        out = torch.min(out, torch.scatter(src, dim, expanded_index, src))
    elif reduce == "max":
        initial_val = float("-inf")
        out.fill_(initial_val)
        out = torch.max(out, torch.scatter(src, dim, expanded_index, src))

    return out
