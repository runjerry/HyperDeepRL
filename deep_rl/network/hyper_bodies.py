#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .hypernetwork_ops import *
from ..utils.hypernet_bodies_defs import *
import numpy as np

class ToyFCHyperBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(ToyFCHyperBody, self).__init__()
        self.mixer = False
        dims = (state_dim,) + hidden_units
        self.config = ToyFCBody_config(state_dim, hidden_units, gate)
        self.gate = gate
        self.feature_dim = dims[-1]
        n_layers = self.config['n_gen']
        self.layers = nn.ModuleList([LinearGenerator(self.config['fc{}'.format(i+1)]).cuda() for i in range(n_layers)])

    def forward(self, x=None, z=None, theta=None):
        if x is None:
            weights = []
            for i, layer in enumerate(self.layers):
                w, b = layer(z[i])
                weights.append(w)
                weights.append(b)
            return weights
        x = x.unsqueeze(0).repeat(z.shape[1], 1, 1)
        for i, layer in enumerate(self.layers):
            if theta:
                x = self.gate(layer(z[i], x, theta[i*2:(i*2)+2]))
            else:
                x = self.gate(layer(z[i], x))
        return x


class DummyHyperBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyHyperBody, self).__init__()
        self.feature_dim = state_dim
        self.config = Dummy_config()

    def forward(self, x=None, z=None):
        if x is None:
            return []
        return x
