import torch as pt
import numpy as np
import H_model
import dynaimic.QuadDynamic as QD
import math


class P_model(pt.nn.Module):
    layers = []

    def __init__(self, arch: list, activation=pt.nn.Tanh, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers.append(pt.nn.Linear(13, arch[0]))
        for i in range(1, len(arch)):
            self.layers.append(activation())
            self.layers.append(pt.nn.Linear(arch[i - 1], arch[i]))
        self.layers.append(pt.nn.Linear(arch[-1], 13))
        self.net = pt.nn.Sequential(*self.layers)
        self.optimizer = pt.optim.Adam(self.net.parameters(), lr=1e-2)

    def forward(self, input: pt.Tensor) -> pt.Tensor:
        out = input
        for i in self.layers:
            out = i(out)
        return out

    def save(self, text: str) -> None:
        pt.save(self.net.state_dict(), text)

    def load(self, text: str) -> None:
        self.net.load_state_dict(pt.load(text))

    def P_loss(self, input: pt.Tensor, output: pt.Tensor, h_model: H_model.H_model, arg=None) -> pt.Tensor:
        if arg is None:
            arg = [1e-1, 1e-1, 1e-3]
        u_ = QD.QuadroCopter().f_u(input, output)
        first = (QD.QuadroCopter().f(input, u_) - h_model(pt.cat([input, output])))
        second = (output - pt.Tensor(QD.QuadroCopter().nabla_g(input)))

        third = pt.abs(( pt.abs(input[:, :3]) @  pt.abs((input[:, :3].T)) ))+ 1 / 3 * pt.abs((pt.abs(u_[:, :, 0]) @ pt.abs(u_[:, :, 0].T)))

        return arg[1] * pt.abs((pt.abs(second) @ pt.abs(second.T))) + arg[2] * pt.abs(third)

    def train_step(self, input: pt.Tensor, H: H_model.H_model) -> float:
        out = self.forward(input)
        loss = 0
        for i in range(35):
            loss += self.P_loss(input=input, output=out, h_model=H)
            u_ = QD.QuadroCopter().f_u(input, out)
            input = QD.QuadroCopter().f(input, u_) * 0.1 + input
            out = self.forward(input)
        loss = self.P_loss(input, out, H)
        if (True):
            print(input[:, :3])
        self.optimizer.zero_grad()
        H.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        H.optimizer.step()
        return (loss @ loss)[0]

    def fit(self, input: pt.Tensor, epochs: int, H: H_model.H_model, verbose=100) -> None:
        for i in range(epochs):
            loss = self.train_step(input, H)
            if i % verbose == 0:
                print(loss)


P = P_model([100, 100])
H = H_model.H_model([100, 100])
sample = pt.Tensor([[10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
P.fit(sample, 100, H, 10)
P.save("www")
