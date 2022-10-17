from torch import nn

def fc_nn_generator(obs_space_num: int,
                    action_space_num: int,
                    hidden_layer_size: int,
                    num_layers: int):
    layers = []
    layers.append(nn.Linear(obs_space_num, hidden_layer_size))
    layers.append(nn.ReLU(inplace=True))
    for _ in range(num_layers):
        layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_layer_size, action_space_num))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
