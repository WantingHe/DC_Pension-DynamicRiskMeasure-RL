"""
Models -- Neural Networks
Policy and value function with fully-connected ANNs

"""
# numpy
import numpy as np
# pytorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# misc
import pdb  # use with set_trace() for the debugger


# build a fully-connected neural net for the policy
class PolicyApprox(nn.Module):
    # constructor
    def __init__(self, input_size, env, n_layers, hidden_size, learn_rate=0.01):
        super(PolicyApprox, self).__init__()
        # input arguments
        self.input_size = input_size
        self.output_size = 5
        self.env = env
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.learn_rate = learn_rate

        # batch normalization for the input
        self.bn_input = nn.BatchNorm1d(self.input_size)

        # build all layers
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size)
        self.layer4 = nn.Linear(self.hidden_size, self.output_size)

        # initializers for weights and biases
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.normal_(self.layer1.bias, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.normal_(self.layer2.bias, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.normal_(self.layer3.bias, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.layer4.weight)
        nn.init.normal_(self.layer4.bias, mean=0.0, std=0.01)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)  # SGD or Adam
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # forward propagation
    def forward(self, x, purchase_constraint=1000):
        epsilon = 1e-3

        loc = F.silu(self.layer1(x))
        loc = F.silu(self.layer2(loc))
        loc = F.silu(self.layer3(loc))

        A_mean = T.sigmoid(self.layer4(loc)[:, :, 0]) * (purchase_constraint.to(self.device))
        I_mean = T.sigmoid(self.layer4(loc)[:, :, 1]) * (purchase_constraint.to(self.device) - A_mean)
        A_std = T.maximum(T.sigmoid(self.layer4(loc)[:, :, 2]) * 0.2 * A_mean,
                          epsilon * T.ones(A_mean.shape, device=self.device)).to(self.device)
        I_std = T.maximum(T.sigmoid(self.layer4(loc)[:, :, 3]) * 0.2 * I_mean,
                          epsilon * T.ones(I_mean.shape, device=self.device)).to(self.device)
        correlation = F.tanh(self.layer4(loc)[:, :, 4]) * (1 - epsilon)

        return A_mean, A_std, I_mean, I_std, correlation


# build a fully-connected neural net for the value function
class ValueApprox(nn.Module):
    # constructor
    def __init__(self, input_size, env, n_layers, hidden_size, learn_rate=0.01):
        super(ValueApprox, self).__init__()
        # input arguments
        self.input_size = input_size
        self.output_size = 1
        self.env = env
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.learn_rate = learn_rate

        self.bn_input = nn.BatchNorm1d(self.input_size)
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size)
        self.layer4 = nn.Linear(self.hidden_size, self.output_size)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # forward propagation
    def forward(self, x):

        x = F.silu(self.layer1(x))
        x = F.silu(self.layer2(x))
        x = F.silu(self.layer3(x))
        x = self.layer4(x)

        return x
