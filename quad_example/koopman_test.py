#!/usr/bin/env python3

import numpy as np
# from stable_koopman_operator import StableKoopmanOperator
from koopman_operator import KoopmanOperator
from quad import Quad
from task import Task, Adjoint
import matplotlib.pyplot as plt
import scipy.linalg
from group_theory import VecTose3, TransToRp, RpToTrans
from lqr import FiniteHorizonLQR
from quatmath import euler2mat

from replay_buffer import ReplayBuffer

import pickle as pkl
from datetime import datetime

import scipy.io as sio
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=20)
parser.add_argument('--type', type=int, default=0)

args = parser.parse_args()

np.set_printoptions(precision=4, suppress=True)
# np.random.seed(50) ### set the seed for reproducibility

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
    koopman_operator = KoopmanOperator(quad.time_step)

    if args.type == 0:
        K = pkl.load(open('al_k_opt.pkl', 'rb'))[-1].T
    elif args.type == 1:
        data = sio.loadmat('Quadrotor_ActiveLearning_Stable_Kd.mat', squeeze_me=True)
        K = data['Kd']

    koopman_operator.set_operator(K)

    adjoint = Adjoint(quad.time_step)

    task = Task() ### this creates the task

    simulation_time = 2000
    horizon = args.T ### time horizon
    sat_val = 6.0 ### saturation value
    control_reg = np.diag([1.] * 4) ### control regularization
    inv_control_reg = np.linalg.inv(control_reg) ### precompute this
    default_action = lambda x : np.random.uniform(-0.1, 0.1, size=(4,)) ### in case lqr control returns NAN

    _R = euler2mat(np.random.uniform(-1.,1., size=(3,)))
    _p = np.array([0., 0., 0.])
    _g = RpToTrans(_R, _p).ravel()
    _twist = np.random.uniform(-0.6, +0.6, size=(6,))
    state = np.r_[_g, _twist]

    target_orientation = np.array([0., 0., -9.81])
    task.inf_weight = 0.

    err = []

    for t in range(simulation_time):

        #### measure state and transform through koopman observables
        m_state = get_measurement(state)

        t_state = koopman_operator.transform_state(m_state)
        err.append(
            np.linalg.norm(m_state[:3] - target_orientation) + np.linalg.norm(m_state[3:])
        )
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

        state = next_state ### backend : update the simulator state
        ### we can also use a decaying weight on inf gain

        if t % 100 == 0:
            print('time : {}, pose : {}, {}'.format(t*quad.time_step,
                                                    get_measurement(state), ustar))
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    if args.type == 0:
        kind = 'actively_learned_koopman'
    elif args.type == 1:
        kind = 'stable_koopman'


    path = './data/'

    if os.path.exists(path) is False:
        os.makedirs(path)
    save_data = {
        'err' : err,
        'kind' : kind,
        'T' : horizon
    }
    pkl.dump(save_data, open(path + 'data_' + date_str + '.pkl', 'wb'))

if __name__=='__main__':
    main()
