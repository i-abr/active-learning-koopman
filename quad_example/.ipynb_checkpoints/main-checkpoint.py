#!/usr/bin/env python3

import numpy as np
from stable_koopman_operator import StableKoopmanOperator
from quad import Quad
from task import Task, Adjoint
import matplotlib.pyplot as plt
import scipy.linalg
from group_theory import VecTose3, TransToRp, RpToTrans
from lqr import FiniteHorizonLQR
from quatmath import euler2mat 

from replay_buffer import ReplayBuffer

np.set_printoptions(precision=4, suppress=True)
np.random.seed(50) ### set the seed for reproducibility

def get_measurement(x):
    g = x[0:16].reshape((4,4)) ## SE(3) matrix
    R,p = TransToRp(g)
    twist = x[16:]
    grot = np.dot(R, [0., 0., -9.81]) ## gravity vec rel to body frame
    return np.concatenate((grot, twist)) 

def get_position(x):
    g = x[0:16].reshape((4,4))
    R,p = TransToRp(g)
    return p

def main():

    quad = Quad() ### instantiate a quadcopter
    replay_buffer = ReplayBuffer(100000)

    ### the timing parameters for the quad are used
    ### to construct the koopman operator and the adjoint dif-eq 
    koopman_operator = StableKoopmanOperator(quad.time_step)
    adjoint = Adjoint(quad.time_step)

    task = Task() ### this creates the task

    simulation_time = 1000
    horizon = 20 ### time horizon 
    sat_val = 6.0 ### saturation value     
    control_reg = np.diag([1.] * 4) ### control regularization 
    inv_control_reg = np.linalg.inv(control_reg) ### precompute this 
    default_action = lambda x : np.random.uniform(-0.1, 0.1, size=(4,)) ### in case lqr control returns NAN


    ### initialize the state 
    #_R = np.diag([1.0, 1.0, 1.0])
    _R = euler2mat(np.random.uniform(-1.,1., size=(3,)))
    _p = np.array([0., 0., 0.])
    _g = RpToTrans(_R, _p).ravel()
    _twist = np.random.uniform(-1., 1., size=(6,)) #* 2.0
    state = np.r_[_g, _twist]

    target_orientation = np.array([0., 0., -9.81])

    err = np.zeros(simulation_time)
    batch_size = 2
    for t in range(simulation_time):

        #### measure state and transform through koopman observables
        m_state = get_measurement(state)

        t_state = koopman_operator.transform_state(m_state)
        err[t] = np.linalg.norm(m_state[:3] - target_orientation) + np.linalg.norm(m_state[3:])

        Kx, Ku = koopman_operator.get_linearization() ### grab the linear matrices
        lqr_policy = FiniteHorizonLQR(Kx, Ku, task.Q, task.R, task.Qf, horizon=horizon) # instantiate a lqr controller
        lqr_policy.set_target_state(task.target_expanded_state) ## set target state to koopman observable state
        lqr_policy.sat_val = sat_val ### set the saturation value 

        ### forward sim the koopman dynamical system (here fdx, fdu is just Kx, Ku in a list)
        trajectory, fdx, fdu, action_schedule = koopman_operator.simulate(t_state, horizon,
                                                                                policy=lqr_policy)
        ldx, ldu = task.get_linearization_from_trajectory(trajectory, action_schedule)
        mudx = lqr_policy.get_linearization_from_trajectory(trajectory)

        rhof = task.mdx(trajectory[-1]) ### get terminal condition for adjoint
        rho = adjoint.simulate_adjoint(rhof, ldx, ldu, fdx, fdu, mudx, horizon)

        ustar = -np.dot(inv_control_reg, fdu[0].T.dot(rho[0])) + lqr_policy(t_state)
        ustar = np.clip(ustar, -sat_val, sat_val) ### saturate control

        if np.isnan(ustar).any():
            ustar = default_action(None)

        ### advacne quad subject to ustar 
        next_state = quad.step(state, ustar)
        
        ### update the koopman operator from data 
        replay_buffer.push(get_measurement(state), ustar, get_measurement(next_state))
        
        if len(replay_buffer) > batch_size:
            input_data, control_data, output_data = replay_buffer.sample(batch_size)
            if t % 100 == 0:
                koopman_operator.compute_operator_from_data(input_data, control_data, 
                                                                output_data, verbose=True)

        state = next_state ### backend : update the simulator state 
        ### we can also use a decaying weight on inf gain
        task.inf_weight = 100.0 * (0.99**(t))
        if t % 100 == 0:
            print('time : {}, pose : {}, {}'.format(t*quad.time_step, 
                                                    get_measurement(state), ustar))

    t = [i * quad.time_step for i in range(simulation_time)]
    plt.plot(t, err)
    plt.xlabel('t')
    plt.ylabel('tracking error')
    plt.show()


if __name__=='__main__':
    main()
