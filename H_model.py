import numpy as np
import matplotlib.pyplot as plt
import torch as pt


class H_model(pt.nn.Module):
    layers = []

    def __init__(self, arch: list, activation=pt.nn.Tanh, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers.append(pt.nn.Linear(13, arch[0]))
        for i in range(1, len(arch)):
            self.layers.append(activation())
            self.layers.append(pt.nn.Linear(arch[i - 1], arch[i]))
        self.layers.append(pt.nn.Linear(arch[-1], 13))
        self.net = pt.nn.Sequential(*self.layers)
        self.optimizer = pt.optim.Adam(self.net.parameters(), lr=1e-3)

    def forward(self, input: pt.Tensor) -> pt.Tensor:
        out = input
        for i in self.layers:
            out = i(out)
        return out

