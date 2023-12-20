### Continuous version for Hamiltonian dynamics training of environments in OpenAI Gym classical controls
### The rendering part is the same as OpenAI Gym
import casadi
import numpy as np
import math
from os import path
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
        self.seed()

        # Viewer for rendering image
        self.viewer = None

    # Dynamics f
    def f(self, q, u):
        return np.zeros((q.shape[0], self.q_dim))

    # Partial derivative of dynamics f wrt control u. Assuming linear control
    def f_u(self, q):
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
    def __init__(self, q_dim=13, u_dim=4, control_coef=0.5, goal_position=np.zeros(shape=(3))):
        super().__init__(q_dim, u_dim, control_coef)
        self.physic = get_quadro_copter_physic()
        self.goal_position = goal_position

    # f(q, u) = dq / dt
    def f(self, q, u):
        qn = self.physic.next_step(q, u)
        return qn - q / 0.1

    # f_u = (df / du) dot p (u0 с крышечкой)
    # np.array(shape)
    def f_u(self, q):
        dfu = casadi.jacobian(self.physic.dyn, self.physic.control)
        dfu_f = casadi.Function('dfu', [self.physic.state, self.physic.control], [dfu])
        arr = np.zeros(shape=(13, 4))
        for i in range(13 * 4):
            arr[i // 4][i % 4] = dfu_f(q, np.array([0, 0, 0, 0]))[i]
        return arr

    # g(q) (награда в конечном состоянии)
    def g(self, q):
        darr = np.array([q[0], q[1], q[2]]) - self.goal_position
        return np.dot(darr, darr)

    # выдает градиенты
    def nabla_g(self, q):
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.g(q + self.eps * self.id[i]) - self.g(q - self.eps * self.id[i])) / (2 * self.eps)
        return ret

    # выдает num рандомных состояний  (квадрокоптер всегда параллельно плоскости и с нулевой скоростью)
    def sample_q(self, num_examples, mode='train'):
        q = []
        ini_v_I = [0.0, 0.0, 0.0]
        ini_q = toQuaternion(0, [1, -1, 1])
        ini_w = [0.0, 0.0, 0.0]
        for i in range(num_examples):
            ini_r_I = [random.randint(-5, 5) for _ in range(3)]
            ini_state = ini_r_I + ini_v_I + ini_q + ini_w
            q.append(ini_state)
        return np.array(q)
