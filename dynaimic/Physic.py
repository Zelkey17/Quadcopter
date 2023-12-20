import casadi
import numpy as np
from Quadrotor import Quadrotor


class QuadroCopterPhysic:
    def __init__(self, state, control, ode, dt):
        self.q_state = 13
        self.u_state = 4
        self.dt = dt
        self.state = state
        self.control = control
        self.dyn = ode
        self.dyn_fn = casadi.Function('dynFun', [self.state, self.control], [self.dyn])

    # поиск траектории по массиву control_val
    # ini_state - начальное состояние
    # horizon - кол во кадров
    # control_val - двумерный массив размера horizon, хранит 4 числа - управление моторами квадрокоптера
    def integrate_sys(self, ini_state, horizon, control_fn):
        state_traj = np.zeros((horizon + 1, self.q_state))
        state_traj[0, :] = ini_state
        for t in range(horizon):
            curr_x = state_traj[t, :]
            curr_u = np.array(control_fn(curr_x))

            state_traj[t + 1, :] = self.dyn_fn(curr_x, curr_u).full().flatten()
        return state_traj

    # то же самое что и integrate_sys, только для 1 шага
    # control_val - массив размера 4 - управление моторами квадрокоптера на текущем шаге
    def next_step(self, state, control_val):
        new_state = self.dyn_fn(state, np.array(control_val)).full().flatten()

        return new_state


def get_quadro_copter_physic():
    uav = Quadrotor()
    Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
    uav.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
    wr, wv, wq, ww = 1, 1, 5, 1
    uav.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)

    dt = 0.1
    dyn = uav.X + dt * uav.f
    qcp = QuadroCopterPhysic(uav.X, uav.U, dyn, dt)
    return qcp
