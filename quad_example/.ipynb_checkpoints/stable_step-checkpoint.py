import numpy as np
from numpy import sin, cos
from scipy.linalg import polar, pinv2, logm
import math

def projectPSD(Q, epsilon = 0, delta = math.inf):
    # if len(locals() >= 3
    # delta = math.inf
    Q = (Q+Q.T)/2
    [e, V] = np.linalg.eig(Q)
    # print(np.diag( np.minimum( delta, np.maximum(e, epsilon) ) ))
    Q_PSD = V.dot(np.diag( np.minimum( delta, np.maximum(e, epsilon) ) )).dot(V.T)
    return Q_PSD

def gradients(X,Y, S, U, B):
    Sinv = np.linalg.inv(S)
    R = Sinv.dot(U).dot(B).dot(S).dot(X)
    # R = np.linalg.multi_dot([Sinv, U, B, S, X])
    # print(R.shape)
    e = np.linalg.norm(Y - R, 'fro')**2

    S_grad = - Sinv.T.dot(R-Y).dot(X.T).dot(S.T).dot(B.T).dot(U.T).dot(Sinv.T) + B.T.dot(U.T).dot(Sinv.T).dot(R-Y).dot(X.T)
    U_grad = - Sinv.T.dot(Y - Sinv.dot(U).dot(B).dot(S).dot(X)).dot(X.T).dot(S.T).dot(B.T)
    B_grad = - U.T.dot(Sinv.T).dot(Y-Sinv.dot(U).dot(B).dot(S).dot(X)).dot(X.T).dot(S.T)

    return e, S_grad, U_grad, B_grad


def stabilize_discrete(X, Y, S, U, B, max_iter=10):
    # n = length()
    n = len(X) # number of Koopman basis functions
    na2 = np.linalg.norm(Y, 'fro')**2

    # Initialization of S, U, and B
    #S = np.identity(n)
    #[U, B] = polar(np.matmul(Y, pinv2(X)) )
    #B = projectPSD(B, 0, 1)

    # parameters
    alpha0 = 0.5 # parameter of FGM
    lsparam = 1.5 # parameter; has to be larger than 1 for convergence
    lsitermax = 20
    gradient = 0 # 1 for standard Gradient Descent; 0 for FGM
    if np.linalg.cond(S) > 1e12 :
        print(" Initial S is ill-conditioned")

    # initial step length: 1/L
    eS,_ = np.linalg.eig(S)
    L = (np.max(eS)/ np.min(eS))**2

    # Initialization
    error,_,_,_ = gradients(X,Y,S,U,B)
    #print("Error is ", error)
    step = 1/L
    i = 1
    alpha0 = 0.5
    alpha = alpha0
    Ys = S.copy()
    Yu = U.copy()
    Yb = B.copy()
    restarti = 1

    while i < max_iter:
        # compute gradient

        _, gS, gU, gB = gradients(X,Y,S,U,B)
        error_next = math.inf
        inner_iter = 1
        step = step * 2

        # print("This is error", error, " at iteration: ", i)

        # Line Search
        while ( (error_next > error) and (  ((i == 1) and (inner_iter <= 100)) or (inner_iter <= lsitermax) ) ):
            Sn = Ys - gS*step
            Un = Yu - gU*step
            Bn = Yb - gB*step

            # Project onto feasible set
            Sn = projectPSD(Sn, 1e-14)
            Un,_ = polar(Un)
            Bn = projectPSD(Bn, 0, 1)
            # print("Projected")
            # print(Sn)
            error_next,_,_,_ = gradients(X,Y, Sn, Un, Bn)
            step = step / lsparam
            inner_iter = inner_iter + 1
            # print(inner_iter)
        if (i == 1):
            inner_iter0 = inner_iter

        # Conjugate with FGM weights, if cost decreased; else, restart FGM
        alpha_next = (math.sqrt(alpha**4 + 4*alpha**2) - alpha**2 )/2
        beta = alpha * (1 - alpha) / (alpha**2 + alpha_next)

        if (inner_iter >= lsitermax + 1): # line search failed
            if restarti == 1:
            # Restart FGM if not a descent direction
                restarti = 0
                alpha_next = alpha0
                Ys = S.copy()
                Yu = U.copy()
                Yb = B.copy()
                error_next = error
                #print(" No descent: Restart FGM")

                # Reinitialize step length
                eS,_ = np.linalg.eig(S)
                L = (np.max(eS)/ np.min(eS))**2
                # Use information from the first step: how many steps to decrease
                step = 1/L/lsparam**inner_iter0
            elif (restarti == 0): # no previous restart/descent direction
                error_next = error
                break
        else:
            #print('dose updated')
            restarti = 1
            if (gradient == 1):
                beta = 0
            Ys = Sn + beta * (Sn - S)
            Yu = Un + beta * (Un - U)
            Yb = Bn + beta * (Bn - B)
            # Keep new iterates in memory
            S = Sn
            U = Un
            B = Bn
            i = i + 1
        error = error_next
        alpha = alpha_next
        print(error) 
        # Check if error is small (1e-6 relative error)
        if (error < 1e-12*na2):
            print("The algorithm converged")
            break
    Kd = np.linalg.inv(S).dot(U).dot(B).dot(S)
    return S, U, B, Kd, error
