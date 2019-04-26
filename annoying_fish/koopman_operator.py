import autograd.numpy as np
from autograd.numpy import sin, cos
from autograd.scipy.linalg import logm

NUM_STATE_OBS_ = 10 # 4 states 20 basis 
NUM_ACTION_OBS_ = 2
NUM_OBS_ = NUM_STATE_OBS_ + NUM_ACTION_OBS_

np.random.seed(10)
h1 = 20 
#W1 = np.random.uniform(-1.0, 1.0, size=(NUM_STATE_OBS_, 3))
#b1 = np.random.uniform(-1.0, 1.0, size=(NUM_STATE_OBS_,))
W1 = np.random.normal(0., 1.0, size=(h1, 3))
b1 = np.random.normal(0., 1.0, size=(h1,))
W2 = np.random.normal(0., 1.0, size=(NUM_STATE_OBS_, h1))
b2 = np.random.normal(0., 1.0, size=(NUM_STATE_OBS_,))
def psix(x):
    z = np.dot(W1, x) + b1
    #return np.sin(z)
    z = np.sin(z)
    #return z
    z = np.dot(W2, z) + b2
    return np.sin(z)
    #return np.concatenate((x, z))

def psiu(x):
    return np.array([
            [1., 0.],
            [0., 1.]])


class KoopmanOperator(object):

    def __init__(self, sampling_time, noise=1.0):
        self.noise = noise 
        self.sampling_time = sampling_time
        self.A = np.zeros((NUM_OBS_,NUM_OBS_))
        self.G = np.zeros(self.A.shape)
        self.K = np.zeros(self.A.shape)
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

    def compute_operator_from_data(self, datain, cdata, dataout):
        # for i in range(len(cdata)-1):
        self.counter += 1
        fk = np.hstack((
                    psix(datain),  np.dot(psiu(datain), cdata)
                    ))
        fkpo = np.hstack((
                    psix(dataout),  np.dot(psiu(dataout), cdata)
                    ))


        self.G = self.G + (np.outer(fk, fk) - self.G)/self.counter
        #self.A = self.A + (np.outer( fk, fkpo)- self.A)/self.counter
        self.A = self.A + (np.outer(fkpo, fk)- self.A)/self.counter
        # self.G = self.G + np.outer(fk, fk)
        # self.A = self.A + np.outer(fk, fkpo)
        #self.K = np.linalg.pinv(self.G + 1e-3 * np.eye(NUM_OBS_)).dot(self.A)
        self.K = np.dot(self.A, np.linalg.pinv(self.G))
        Kcont = np.real(logm(self.K, disp=False)[0]/self.sampling_time)
        self.Kx = Kcont[0:NUM_STATE_OBS_, 0:NUM_STATE_OBS_]
        self.Ku = Kcont[0:NUM_STATE_OBS_, NUM_STATE_OBS_:NUM_OBS_]
        #self.Kx = Kcont.T[0:NUM_STATE_OBS_, 0:NUM_STATE_OBS_]
        #self.Ku = Kcont.T[0:NUM_STATE_OBS_, NUM_STATE_OBS_:NUM_OBS_]

        #try:
        #    self.K = np.linalg.pinv(self.G).dot(self.A)
        #    Kcont = np.real(logm(self.K)/self.sampling_time)
        #    self.Kx = Kcont.T[0:NUM_STATE_OBS_, 0:NUM_STATE_OBS_]
        #    self.Ku = Kcont.T[0:NUM_STATE_OBS_, NUM_STATE_OBS_:NUM_OBS_]
        #except np.LinAlgError as e:
        #    print('did not invert')

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
