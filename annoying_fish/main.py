#!/usr/bin/env python3

import numpy as np
from koopman_operator import KoopmanOperator
from fish import Fish
from task import Task, Adjoint
import matplotlib.pyplot as plt
import scipy.linalg
from group_theory import VecTose3, TransToRp, RpToTrans
from lqr import FiniteHorizonLQR
from quatmath import euler2mat 

np.set_printoptions(precision=4, suppress=True)
np.random.seed(50) ### set the seed for reproducibility

def get_measurement(x):
    return x

def main():

    fish = Fish() ### instantiate a quadcopter

    ### the timing parameters for the quad are used
    ### to construct the koopman operator and the adjoint dif-eq 
    koopman_operator = KoopmanOperator(fish.time_step)
    adjoint = Adjoint(fish.time_step)

    task = Task() ### this creates the task

    simulation_time = 1000
    horizon = 5 ### time horizon 
    sat_val = 1.0 ### saturation value     
    control_reg = 100*np.diag([1.] * 2) ### control regularization 
    inv_control_reg = np.linalg.inv(control_reg) ### precompute this 
    default_action = lambda x : np.random.uniform(-0.1, 0.1, size=(2,)) ### in case lqr control returns NAN

    for _ in range(1):
        ### initialize the state
        _trajectory = np.zeros((simulation_time, fish.num_states))
        state = np.random.uniform(-0.4, 0.4, size=(fish.num_states, ))
        err = np.zeros(simulation_time)
        task.update_inf_weight(0.)
        for t in range(simulation_time):
            #task.update_inf_weight(100.0 * (0.9**(t)))
            _trajectory[t,:] = state.copy()
            #### measure state and transform through koopman observables
            m_state = get_measurement(state)
            t_state = koopman_operator.transform_state(m_state)
            #err[t] = np.linalg.norm(m_state[:3] - target_orientation) + np.linalg.norm(m_state[3:])

            Kx, Ku = koopman_operator.get_linearization() ### grab the linear matrices
            lqr_policy = FiniteHorizonLQR(Kx, Ku, task.Q, task.R, task.Qf, horizon=horizon, ts=fish.time_step) # instantiate a lqr controller
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
            #ustar = lqr_policy(t_state)
            print(ustar, lqr_policy(t_state))
            ustar = np.clip(ustar, -sat_val, sat_val) + np.random.normal(0., 0.01, size=(fish.num_actions,))### saturate control

            if np.isnan(ustar).any():
                ustar = default_action(None)

            ### advacne quad subject to ustar 
            next_state = fish.step(state, ustar)
            ### update the koopman operator from data 
            koopman_operator.compute_operator_from_data(get_measurement(state),
                                                        ustar, 
                                                        get_measurement(next_state))
            print(np.diag(koopman_operator.K)[:5], next_state, ustar)
            state = next_state ### backend : update the simulator state 
            ### we can also use a decaying weight on inf gain
            #if t % 2 == 0:
            #    print('time : {}, pose : {}, {}, {}'.format(t*fish.time_step, 
            #                                            get_measurement(state), ustar, task.inf_weight))

        t = [i * fish.time_step for i in range(simulation_time)]
        plt.plot(t, _trajectory)
        plt.xlabel('t')
        plt.ylabel('tracking error')
        plt.show()


if __name__=='__main__':
    main()
