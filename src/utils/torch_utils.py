import torch

def get_available_device(debug=False):
    """
    :return: Returns the first available device from the following list: cuda, mps, cpu
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if debug:
        device = torch.device("cpu")

    return device
