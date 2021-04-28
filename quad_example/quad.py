import numpy as np
from numpy import cos, sin, cross
from group_theory import VecTose3, TransToRp, RpToTrans
#from autograd import jacobian

class Quad(object):

    num_states = 22
    num_actions = 4
    time_step = 1.0/200.0
    inertia_matrix = np.diag([0.04, 0.0375, 0.0675])
    inv_inertia_matrix = np.linalg.inv(inertia_matrix)

    kt = 0.6
    km = 0.15
    arm_length = 0.2
    mass = 0.6
    inv_mass = 1.0/mass

    vel_damping = np.array([0.01, 0.01, 0.05])*40
    ang_damping = np.array([0.05, 0.05, 0.01])*40

    def __init__(self):
        pass
        #self.fdx = jacobian(self.f, argnum=0)
        #self.fdu = jacobian(self.f, argnum=1)

    def f(self, x, uu):


        u = np.clip(uu, -6,6)
        g = x[0:16].reshape((4,4))
        R,p = TransToRp(g)

        omega = x[16:19]
        v =  x[19:]
        twist = x[16:]


        # u[0] *= 0.8
        F = self.kt * (u[0] + u[1] + u[2] + u[3])
        M = np.array([
                        self.kt * self.arm_length * (u[1] - u[3]),
                        self.kt * self.arm_length * (u[2] - u[0]),
                        self.km * (u[0] - u[1] + u[2] - u[3])
                        ])

        inertia_dot_omega = np.dot(self.inertia_matrix, omega)
        inner_rot = M + cross(inertia_dot_omega , omega)

        omegadot = np.dot(
                    self.inv_inertia_matrix,
                    inner_rot
                    ) - self.ang_damping * omega

        vdot = self.inv_mass * F * np.array([0.,0.,1.]) - cross(omega, v) - 9.81 * np.dot(R.T,np.array([0.,0.,1.])) - self.vel_damping * v


        dg = np.dot(g, VecTose3(twist)).ravel()

        return np.concatenate((dg, omegadot, vdot))

    def step(self, state, action):
        k1 = self.f(state, action) * self.time_step
        k2 = self.f(state + k1/2.0, action) * self.time_step
        k3 = self.f(state + k2/2.0, action) * self.time_step
        k4 = self.f(state + k3, action) * self.time_step
        return state + (k1 + 2.0 * (k2 + k3) + k4)/6.0
