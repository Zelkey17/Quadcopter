### Continuous version for Hamiltonian dynamics training of environments in OpenAI Gym classical controls
### The rendering part is the same as OpenAI Gym
import casadi
import numpy as np
import math
from os import path

import torch

from Physic import get_quadro_copter_physic
from Quadrotor import toQuaternion, Quadrotor
import random


### Generic continuous environment for reduced Hamiltonian dynamics framework
class ContinuousEnv():
    def __init__(self, q_dim=13, u_dim=4, control_coef=0.5):
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.control_coef = control_coef
        self.eps = 1e-8
        self.id = np.eye(q_dim)

    # Dynamics f
    def f(self, q, u):
        return np.zeros((q.shape[0], self.q_dim))

    # Partial derivative of dynamics f wrt control u. Assuming linear control
    def f_u(self, q, p):
        return np.zeros((q.shape[0], self.q_dim, self.u_dim))

    # Lagrangian or running cost L
    def L(self, q, u):
        return self.control_coef * np.sum(u ** 2, axis=1) + self.g(q)

    # Terminal cost g
    def g(self, q):
        return np.zeros(q.shape[0])

    # Nabla of g
    def nabla_g(self, q):
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.g(q + self.eps * self.id[i]) - self.g(q - self.eps * self.id[i])) / (2 * self.eps)
        return ret

    # Sampling state q
    def sample_q(self, num_examples, mode='train'):
        return np.zeros((num_examples, self.q_dim))


class QuadroCopter(ContinuousEnv):
    def __init__(self, q_dim=13, u_dim=4, control_coef=0.5, goal_position=torch.zeros(3)):
        super().__init__(q_dim, u_dim, control_coef)
        self.physic = get_quadro_copter_physic()
        self.goal_position = goal_position

    # f(q, u) = dq / dt
    def f(self, q: torch.tensor, u: torch.tensor) -> torch.tensor:
        fn = torch.zeros((q.shape[0], 13))
        for i in range(q.shape[0]):
            qn = self.physic.next_step(q[i], u[i])
            fn[i] = (qn - q) / self.physic.dt
        return fn

    # f_u = (df(q) / du) @ p (u0 с крышечкой)
    def f_u(self, q, p):
        u = torch.zeros(size=(q.shape[0], 4))
        q0 = self.f(q, u) + q
        qs = torch.zeros(size=(q.shape[0], 4, 13))
        for i in range(4):
            u[:, i] = 0.01
            qs[:, i, :] = self.f(q, u)
            u[:, i] = 0
            qs[:, i, :] -= q0
        return -(qs / 0.01) @ p.T

    # g(q) (лосс в конечном состоянии)
    def g(self, q):
        darr = q[:, :3] - self.goal_position
        loss = torch.zeros(q.shape[0])
        for i in range(darr.shape[0]):
            loss[i] = torch.dot(darr[i], darr[i])
        return loss

    # выдает градиенты

    # выдает num рандомных состояний  (квадрокоптер всегда параллельно плоскости и с нулевой скоростью)
    def sample_q(self, num_examples, mode='train') -> torch.tensor:
        q = torch.zeros(size=(num_examples, 13))
        ini_v_I = [0.0, 0.0, 0.0]
        ini_q = toQuaternion(0, [1, -1, 1])
        ini_w = [0.0, 0.0, 0.0]
        for i in range(num_examples):
            ini_r_I = [random.randint(-5, 5) for _ in range(3)]
            ini_state = ini_r_I + ini_v_I + ini_q + ini_w
            q[i] = torch.tensor(ini_state)
        return q

    # выдает градиенты g
    def nabla_g(self, q):
        grad = torch.zeros(size=(q.shape[0], 13))
        for i in range(q.shape[0]):
            gr = torch.cat(((2 * q[i, :3] - 2 * self.goal_position), torch.zeros(self.q_dim - 3)))
            grad[i] = gr
        return grad


c = QuadroCopter()
q = c.sample_q(5)
print(q)
print(c.g(q))