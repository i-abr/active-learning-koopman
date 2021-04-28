import numpy as np
import matplotlib.pyplot as plt
from koopman_operator import psix, psiu, NUM_OBS_, NUM_STATE_OBS_, NUM_ACTION_OBS_


class Adjoint(object):

    def __init__(self, sampling_time):
        self.sampling_time = sampling_time

    def rhodt(self, rho, ldx, ldu, fdx, fdu, mudx):
        return - np.dot((fdx + np.dot(fdu, mudx)).T, rho) - (ldx + np.dot(mudx.T, ldu))

    def simulate_adjoint(self, rhof, ldx, ldu, fdx, fdu, mudx, N):

        rho = [None] * N
        rho[N-1] = rhof
        for i in reversed(range(1, N)):
            rhodt = self.rhodt(rho[i], ldx[i-1], ldu[i-1], fdx[i-1], fdu[i-1], mudx[i-1])
            rho[i-1] = rho[i] - rhodt * self.sampling_time
        return rho


class Task(object):


    def __init__(self):
        Qdiag = np.zeros(NUM_STATE_OBS_)
        Qdiag[0:3] = 1. # g vec
        Qdiag[3:6] = 1. # omega
        Qdiag[6:9] = 5. # v
        self.Q = np.diag(Qdiag)*10.0
        Qfdiag = np.ones(NUM_STATE_OBS_)
        Qfdiag[3:] = 10.0
        self.Qf = 0*np.diag(Qfdiag)
        #self.Qf *= 0
        self.target_state = np.zeros(9)
        self.target_state[2] = -9.81
        self.target_expanded_state = psix(self.target_state)
        self.R = np.diag([1.0]*4)
        self.inf_weight = 100.0
        self.eps = 1e-5
        self.final_cost = 0

    def l(self, state, action):
        error = state - self.target_expanded_state
        error_q = np.dot(self.Q, error)
        action_r = np.dot(self.R, action)
        return np.dot(error, error_q) + np.dot(action, action_r) + self.inf_weight / (np.dot(state, state)+self.eps)

    def get_stab_cost(self, state):
        return np.dot(state[3:9], state[3:9])

    def information_gain(self, state):
        return np.dot(state, state)

    def ldx(self, state, action):
        error = state - self.target_expanded_state
        d_err = np.zeros(state.shape)
        d_err = state
        return np.dot(self.Q, error) - self.inf_weight * 2.0 * d_err/ (np.dot(state, state) + self.eps)**2

    def ldu(self, state, action):
        action_r = np.dot(self.R, action)
        return action_r

    def m(self, state):
        error = state - self.target_expanded_state
        error_q = np.dot(self.Qf, error)
        return np.dot(error, error_q)*self.final_cost

    def mdx(self, state):
        error = state - self.target_expanded_state
        return np.dot(self.Qf, error) * self.final_cost

    def get_linearization_from_trajectory(self, trajectory, actions):
        return [ self.ldx(state, action) for state, action in zip(trajectory, actions)], [self.ldu(state, action)  for state, action in zip(trajectory, actions)]

    def trajectory_cost(self, trajectory, actions):
        total_cost = 0.0
        for state, action in zip(trajectory, actions):
            total_cost += self.l(state, action)

        return total_cost + self.m(trajectory[-1])
