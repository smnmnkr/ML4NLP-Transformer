import torch


#
#
#  -------- get_device -----------
#
def get_device() -> torch.device:
    """
    Returns best possible computing device.

    :return: string (cuda, cpu)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
#  -------- get_tgt_mask -----------
#
def get_tgt_mask(size: int) -> torch.tensor:
    """
    Generates a square matrix where the each row allows one token more to be seen

    :param size: int
    :return: torch.tensor
    """

    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return mask
