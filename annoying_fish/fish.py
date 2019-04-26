import autograd.numpy as np
from autograd.numpy import cos, sin, cross, pi
from group_theory import VecTose3, TransToRp, RpToTrans
from autograd import jacobian

m_b  = 0.725 
m_ax = -0.217
m_ay = -0.7888
J_bz = 2.66 * 10**-3
J_az = -7.93 * 10**-4
d    = 0.04
rho  = 1000.0
S    = 0.03 
C_D  = 0.97            
C_L  = 3.9047          
K_D  = 4.5 * 10**-3

omega_a       = 2*pi
m             = pi/4. * rho * d**2 
m1            = m_b - m_ax
m2            = m_b - m_ay
J3            = J_bz - J_az
c             = 0.105
c1            = 0.5 * rho * S * C_D
c2            = 0.5 * rho * S * C_L
c4            = 1./ J3 * K_D
L             = 0.071
Kf            = 0.7 
Km            = 0.45


class Fish(object):
    def __init__(self):
        self.num_states = 3
        self.num_actions = 2
        self.time_step = 1.0/50.0

    def f(self, x, u):
        #u1 = u[0]
        #u2 = u[1]
        u1 = np.clip(u[0], -0.8, 0.8) + 0.8
        u2 = np.clip(u[1], -0.25, 0.25)

        v1 = x[0]
        v2 = x[1] 
        omega = x[2]

        if np.abs(v2) < 1e-12 and np.abs(v1) < 1e-12:
            atanv1v2 = 0; # 0/0 gives NaN
        else:
            atanv1v2 = np.arctan(v2/v1);

        f = np.array([
                (12*m2*omega*v2 - 12*c1*v1*(v1**2 + v2**2)**(0.5) + 12*c2*v2*atanv1v2*(v1**2 + v2**2)**(0.5) + Kf*L**2*m*omega_a**2*u1)/(12*m1),
                -(4*c1*v2*(v1**2 + v2**2)**(0.5) + 4*m1*omega*v1 + 4*c2*v1*atanv1v2*(v1**2 + v2**2)**(0.5) - Kf*L**2*m*omega_a**2*u2)/(4*m2),
                v1*v2*(m1 - m2) - c4*omega**2*np.sign(omega) - (Km*L**2*c*m*omega_a**2*u2)/(4*J3)
        ])
        return f

    def step(self, state, action):
        k1 = self.f(state, action) * self.time_step
        k2 = self.f(state + k1/2.0, action) * self.time_step
        k3 = self.f(state + k2/2.0, action) * self.time_step
        k4 = self.f(state + k3, action) * self.time_step
        return state + (k1 + 2.0 * (k2 + k3) + k4)/6.0



if __name__=='__main__':
    fish = Fish()
    x = np.array([0.1, 0.2, 0.3, 0.4])
    u = np.array([0.1, 0.3])
    print(fish.step(x, u))
