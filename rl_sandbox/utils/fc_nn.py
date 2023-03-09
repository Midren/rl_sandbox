import typing as t
from torch import nn

def fc_nn_generator(input_num: int,
                    output_num: int,
                    hidden_size: int,
                    num_layers: int,
                    intermediate_activation: t.Type[nn.Module] = nn.ReLU,
                    final_activation: nn.Module = nn.Identity(),
                    layer_norm: bool = False):
    norm_layer = nn.Identity if layer_norm else nn.LayerNorm
    assert num_layers >= 3
    layers = []
    layers.append(nn.Linear(input_num, hidden_size))
    layers.append(nn.LayerNorm(hidden_size))
    layers.append(intermediate_activation(inplace=True))
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(norm_layer(hidden_size))
        layers.append(intermediate_activation(inplace=True))
    layers.append(nn.Linear(hidden_size, output_num))
    layers.append(final_activation)
    return nn.Sequential(*layers)
