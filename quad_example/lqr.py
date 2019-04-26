import numpy as np


class FiniteHorizonLQR(object):


    def __init__(self, A, B, Q, R, F, horizon=10):


        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Rinv = np.linalg.inv(R)
        self.F = F
        self.time_step = 1/200
        self.horizon = horizon
        self.sat_val = 6.0
        self.target_state = None
        self.active = 1
        self.final_cost = 0

    def set_target_state(self, target):
        self.target_state = target

    def get_control_gains(self):
        P = [None] * self.horizon
        P[-1] = self.F.copy()
        K = [None] * (self.horizon)
        r = [None] * (self.horizon)
        K[-1] = self.Rinv.dot(self.B.T.dot(P[-1]))
        r[-1] = self.F.dot(self.target_state*0.0)
        for i in reversed(range(1, self.horizon)):
            PB = np.dot(P[i], self.B)
            BP = np.dot(self.B.T, P[i])
            PBRB = np.dot(PB, np.dot(self.Rinv, self.B.T))
            Pdot = - (np.dot(self.A.T, P[i]) + np.dot(P[i], self.A) - np.dot(np.dot(PB, self.Rinv), BP) + self.Q)
            rdot = -(self.A.T.dot(r[i]) - self.Q.dot(self.target_state) - PBRB.dot(r[i]))
            P[i-1] = P[i] - Pdot*self.time_step
            K[i-1] = self.Rinv.dot(self.B.T.dot(P[i]))
            r[i-1] = r[i] - rdot*self.time_step
        return K, r

    def __call__(self, state):
        K,r = self.get_control_gains()
        ref = -self.Rinv.dot(self.B.T).dot(r[0])
        return np.clip(-K[0].dot(state-self.target_state), -self.sat_val, self.sat_val)
    
    def get_linearization_from_trajectory(self, trajectory):
        K,_ = self.get_control_gains()
        return [-k for k in K]
