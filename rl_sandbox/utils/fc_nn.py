import typing as t
from torch import nn

def fc_nn_generator(input_num: int,
                    output_num: int,
                    hidden_size: int,
                    num_layers: int,
                    final_activation: t.Type[nn.Module] = nn.Identity):
    layers = []
    layers.append(nn.Linear(input_num, hidden_size))
    layers.append(nn.ReLU(inplace=True))
    for _ in range(num_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_size, output_num))
    layers.append(final_activation())
    return nn.Sequential(*layers)
