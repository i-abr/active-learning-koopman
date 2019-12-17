import numpy as np
from numpy import sin, cos
from scipy.linalg import polar, pinv2, logm

from stable_step import stabilize_discrete, projectPSD, gradients
from copy import deepcopy

NUM_STATE_OBS_ = 18
NUM_ACTION_OBS_ = 4
NUM_OBS_ = NUM_STATE_OBS_ + NUM_ACTION_OBS_

def psix(x):
    w1 = x[3];
    w2 = x[4];
    w3 = x[5];
    v1 = x[6];
    v2 = x[7];
    v3 = x[8];
    return np.array([
          x[0],x[1],x[2], # g
          x[3],x[4],x[5], # omega
          x[6],x[7],x[8], # v
          v3 * w2,
          v2 * w3,
          v3 * w1,
          v1 * w3,
          v2 * w1,
          v1 * w2,
          w2 * w3,
          w1 * w3,
          w1 * w2,
            ])

def psiu(x):
    return np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.,],
        [0., 0., 1., 0.,],
        [0., 0., 0., 1.,]])



class StableKoopmanOperator(object):

    def __init__(self, sampling_time, noise=1.0):
        self.noise = noise
        self.sampling_time = sampling_time


        ### stable koopman setup
        self.A = np.zeros((NUM_OBS_,NUM_OBS_))
        self.G = np.zeros(self.A.shape)
        self.K = np.zeros(self.A.shape)

        self.alpha = 1e-5

        self.Kx = np.ones((NUM_STATE_OBS_, NUM_STATE_OBS_))
        self.Kx = np.random.normal(0., 1.0, size=self.Kx.shape) * noise
        self.Ku = np.ones((NUM_STATE_OBS_, NUM_ACTION_OBS_))
        self.Ku = np.random.normal(0., 1.0, size=self.Ku.shape) * noise
        self.counter = 0


    def clear_operator(self):
        self.counter = 0
        self.Kx = np.random.normal(0., 1.0, size=self.Kx.shape) * self.noise
        self.Ku = np.random.normal(0., 1.0, size=self.Ku.shape) * self.noise
        self.G = np.zeros(self.A.shape)
        self.K = np.zeros(self.A.shape)

    def compute_operator_from_data(self, datain, cdata, dataout, verbose=False, max_iter=10):
        fk = np.hstack((
                    psix(datain),  np.dot(psiu(datain), cdata)
                    ))
        fkpo = np.hstack((
                    psix(dataout),  np.dot(psiu(dataout), cdata)
                    ))
        self.G = self.G + (np.outer(fk, fk) - self.G)/(self.counter+1)
        self.A = self.A + (np.outer(fkpo, fk)- self.A)/(self.counter+1)

        self.K_lsq = np.dot(self.A, np.linalg.pinv(self.G))

        if self.counter == 0:
            self.S = np.identity(NUM_OBS_)
            [self.U, self.B] = polar(self.K_lsq)
            self.B = projectPSD(self.B, 0, 1)
        [self.U, self.B] = polar(self.K_lsq)
        self.B = projectPSD(self.B, 0, 1)
        Sprev = self.S.copy()
        Uprev = self.U.copy()
        Bprev = self.B.copy()

        _, ds, du, db = gradients(self.G, self.A, self.S, self.U, self.B)
        N = self.counter + 1
        self.S -= 1e-3 * np.real(ds) / N
        self.U -= 1e-3 * np.real(du) / N
        self.B -= 1e-3 * np.real(db) / N
        # self.S, self.U, self.B, self._K_proj, err = stabilize_discrete(deepcopy(self.A), deepcopy(self.G), deepcopy(self.S), deepcopy(self.U), deepcopy(self.B), max_iter=max_iter)
        self._K_proj = np.dot(self.U, self.B)
        # print(np.linalg.norm(self.S-Sprev, 'fro'), np.linalg.norm(self.U-Uprev, 'fro'))

        self.K = (1-0.99) * self._K_proj + 0.99 * self.K_lsq

        print(np.max(np.linalg.eig(self.K)[0]))

        Kcont = np.real(logm(self.K, disp=False)[0]/self.sampling_time)
        self.Kx = Kcont[0:NUM_STATE_OBS_, 0:NUM_STATE_OBS_]
        self.Ku = Kcont[0:NUM_STATE_OBS_, NUM_STATE_OBS_:NUM_OBS_]
        if verbose:
            print(err)
        self.counter += 1

    #def compute_operator_from_data(self, datain, cdata, dataout, verbose=False, max_iter=10):
    #    # X in R n x P
    #    # Y in R n x P, n is num basis
    #    X = []
    #    Y = []
    #    for x, xpo, u in zip(datain, dataout, cdata):
    #        X.append(
    #            np.concatenate([psix(x), np.dot(psiu(x), u)], axis=0)
    #        )
    #        Y.append(
    #            np.concatenate([psix(xpo), np.dot(psiu(xpo), u)], axis=0)
    #        )
    #    X = np.stack(X).T
    #    Y = np.stack(Y).T

    #    assert X.shape[0] == NUM_OBS_, 'Looks like it could be backwards'

    #
    #    if self.counter == 0:
    #        self.S = np.identity(NUM_OBS_)
    #        [self.U, self.B] = polar(np.matmul(Y, pinv2(X)) )
    #        self.B = projectPSD(self.B, 0, 1)
    #    else:
    #        self.S = np.identity(NUM_OBS_)
    #        [self.U, self.B] = polar(np.matmul(Y, pinv2(X)) )
    #        self.B = projectPSD(self.B, 0, 1)

    #    self.S, self.U, self.B, self.K, err = stabilize_discrete(X, Y, self.S.copy(), self.U.copy(), self.B.copy(), max_iter=max_iter)

    #    Kcont = np.real(logm(self.K, disp=False)[0]/self.sampling_time)
    #    self.Kx = Kcont[0:NUM_STATE_OBS_, 0:NUM_STATE_OBS_]
    #    self.Ku = Kcont[0:NUM_STATE_OBS_, NUM_STATE_OBS_:NUM_OBS_]
    #    if verbose:
    #        print(err)
    #
    #    return X, Y

    def transform_state(self, state):
        return psix(state)

    def f(self, state, action):
        # state=psix(state)
        return np.dot(self.Kx, state) + np.dot(self.Ku, action)

    def g(self, state):
        return self.Ku

    def get_linearization(self):
        return self.Kx, self.Ku

    def step(self, state, action):
        return state + self.f(state, action) * self.sampling_time

    # def step(self, state, action):
    #     k1 = self.f(state, action) * self.sampling_time
    #     k2 = self.f(state + k1/2.0, action) * self.sampling_time
    #     k3 = self.f(state + k2/2.0, action) * self.sampling_time
    #     k4 = self.f(state + k3, action) * self.sampling_time
    #     return state + (k1 + 2.0 * (k2 + k3) + k4)/6.0


    def simulate(self, state, N, action_schedule=None, policy=None):

        trajectory = [state.copy()]
        ldx = [self.Kx for  i in range(N)]
        ldu = [self.g(state)]
        actions = []
        for i in range(N-1):
            if policy is not None:
                action = policy(state)
                actions.append(action.copy())
            if action_schedule is not None:
                action = action_schedule[i]
            state = self.step(state, action)
            ldu.append(self.g(state))
            trajectory.append(state.copy())
        return trajectory, ldx, ldu, actions


    def simulate_mixed_policy(self, x0, N, ustar, policy, tau, lam):

        x = [None] * N
        x[0] = x0.copy()
        _u = [None] * (N-1)
        for i in range(N-1):
            if  tau <= i <= tau+lam:
                ueval = ustar
            else:
                ueval = policy(x[i])
            _u[i] = ueval
            x[i+1] = self.step(x[i], ueval)
        return x, _u
