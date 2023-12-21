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
        fn = []
        for i in range(q.shape[0]):
            qn = self.physic.next_step(q[i], u[i])
            fn.append(qn - q / self.physic.dt)
        return torch.tensor(fn)

    # f_u = (df(q) / du) @ p (u0 с крышечкой)
    def f_u(self, q, p):
        dfp = []
        dfu = casadi.jacobian(self.physic.dyn, self.physic.control)
        dfu_f = casadi.Function('dfu', [self.physic.state, self.physic.control], [dfu])
        for i in range(q.shape[0]):
            grad = dfu_f(np.array(q[i]), np.zeros(4))
            arr = torch.zeros(size=(self.q_dim, self.u_dim))
            for j in range(self.q_dim * self.u_dim):
                arr[j // self.u_dim][j % self.u_dim] = float(grad[j])
            print(arr)
            dfp.append(arr.T @ p[i].T)
        return np.array(dfp)

    # g(q) (лосс в конечном состоянии)
    def g(self, q):
        darr = q[:, :3] - self.goal_position
        loss = []
        for i in range(darr.shape[0]):
            loss.append(np.dot(darr[i], darr[i]))
        return np.array(loss)

    # выдает градиенты

    # выдает num рандомных состояний  (квадрокоптер всегда параллельно плоскости и с нулевой скоростью)
    def sample_q(self, num_examples, mode='train') -> torch.tensor:
        q = []
        ini_v_I = [0.0, 0.0, 0.0]
        ini_q = toQuaternion(0, [1, -1, 1])
        ini_w = [0.0, 0.0, 0.0]
        for i in range(num_examples):
            ini_r_I = [random.randint(-5, 5) for _ in range(3)]
            ini_state = ini_r_I + ini_v_I + ini_q + ini_w
            q.append(ini_state)
        return torch.tensor(q)

    # выдает градиенты g
    def nabla_g(self, q):
        grad = []
        for i in range(q.shape[0]):
            gr = np.concatenate(((2 * q[i, :3] - 2 * self.goal_position), np.zeros(shape=(self.q_dim - 3))))
            grad.append(gr)
        return np.array(grad)
