from typing import Optional, List
from pdb import set_trace as stop

import torch.nn as nn


def get_model(
    input_dim: int,
    output_dim: int,
    hidden_layers: Optional[List[int]] = None,
):
    """
    Feed-forward network, made of linear layers with ReLU activation functions
    The number of layers, and their size is given by `hidden_layers`.
    """
    # assert init_method in {'default', 'xavier'}

    if hidden_layers is None:
        # linear model
        model = nn.Sequential(nn.Linear(input_dim, output_dim))

    else:
        # neural network
        # there are hidden layers in this case.
        dims = [input_dim] + hidden_layers + [output_dim]
        modules = []
        for i, dim in enumerate(dims[:-2]):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(dims[-2], dims[-1]))
        model = nn.Sequential(*modules)
        # stop()

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{n_parameters:,} parameters')

    return model

def count_parameters(model: nn.Module) -> int:
    """"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)