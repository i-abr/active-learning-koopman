import numpy as np
from numpy import sin, cos
from scipy.linalg import polar, pinv2, logm
import math



def projectPSD(Q, epsilon = 0, delta = math.inf):
    # if len(locals() >= 3
    # delta = math.inf
    Q = (Q+Q.T)/2.0
    [e, V] = np.linalg.eig(Q)
    # print(np.diag( np.minimum( delta, np.maximum(e, epsilon) ) ))
    Q_PSD = V.dot(np.diag( np.minimum( delta, np.maximum(e, epsilon) ) )).dot(V.T)
    return Q_PSD

#def gradients(X,Y, S, U, B):
def gradients(G, A, S, U, B):
    Sinv = np.linalg.inv(S)
    # TODO: R = Sinv.dot(U).dot(B).dot(S).dot(X)
    R = Sinv.dot(U).dot(B).dot(S)
    e = 0.0#np.linalg.norm(Y - R, 'fro')**2

    S_grad = -Sinv.T.dot(R.dot(G)-A).dot(S.T).dot(B.T).dot(U.T).dot(Sinv.T) \
                + B.T.dot(U.T).dot(Sinv.T).dot(R.dot(G)-A)
    U_grad = -Sinv.T.dot(A-Sinv.dot(U).dot(B).dot(S).dot(G)).dot(S.T).dot(B.T)
    B_grad = -U.T.dot(Sinv.T).dot(A-Sinv.dot(U).dot(B).dot(S).dot(G)).dot(S.T)

    return e, S_grad, U_grad, B_grad


def stabilize_discrete(G, A, S, U, B, max_iter=10):
    # n = length()
    # n = len(A) # number of Koopman basis functions
    # na2 = 10.0#np.linalg.norm(Y, 'fro')**2

    # Initialization of S, U, and B
    #S = np.identity(n)
    #[U, B] = polar(np.matmul(Y, pinv2(X)) )
    #B = projectPSD(B, 0, 1)

    # parameters
    # alpha0 = 0.5 # parameter of FGM
    # lsparam = 1.5 # parameter; has to be larger than 1 for convergence
    # lsitermax = 20
    # gradient = 1 # 1 for standard Gradient Descent; 0 for FGM
    # if np.linalg.cond(S) > 1e12 :
    #     print(" Initial S is ill-conditioned")

    # initial step length: 1/L
    # eS,_ = np.linalg.eig(S)
    # L = (np.max(eS)/ np.min(eS))**2

    # Initialization
    #error,_,_,_ = gradients(G,A,S,U,B)
    #print("Error is ", error)
    # step = 1e-8#1.0/L
    # i = 0
    # alpha0 = 0.5
    # alpha = alpha0
    # Sn = S.copy()
    # Un = U.copy()
    # Bn = B.copy()
    # restarti = 1
    step = 1e-6
    _, gS, gU, gB = gradients(G,A,S,U,B)
    S = S - gS*step
    U = U - gU*step
    B = B - gB*step
    S = projectPSD(S, 0)
    U,_ = polar(U)
    B = projectPSD(B, 0, 1)
    # while i < max_iter:
    #     # compute gradient
    #
    #     _, gS, gU, gB = gradients(G,A,Sn,Un,Bn)
    #     error_next = math.inf
    #     inner_iter = 1
    #     #step = step * 2
    #
    #     # print("This is error", error, " at iteration: ", i)
    #
    #     # Line Search
    #     # while ( (error_next > error) and (  ((i == 1) and (inner_iter <= 100)) or (inner_iter <= lsitermax) ) ):
    #     #while inner_iter <= lsitermax:
    #     Sn = Sn - gS*step
    #     Un = Un - gU*step
    #     Bn = Bn - gB*step
    #
    #     # Project onto feasible set
    #     Sn = projectPSD(Sn, 0)
    #     Un,_ = polar(Un)
    # #     Bn = projectPSD(Bn, 0, 1)
    #     # print("Projected")
    #     # print(Sn)
    #     #error_next,_,_,_ = gradients(G,A, Sn, Un, Bn)
    #     #step = step / lsparam
    #     inner_iter = inner_iter + 1
    #         # print(inner_iter)
    #     #if (i == 1):
    #     #    inner_iter0 = inner_iter
    #
    #     # Conjugate with FGM weights, if cost decreased; else, restart FGM
    #     #alpha_next = (math.sqrt(alpha**4 + 4.*(alpha**2)) - alpha**2 )/2.
    #     #beta = alpha * (1.0 - alpha) / (alpha**2 + alpha_next)
    #
    #     #if (inner_iter >= lsitermax + 1): # line search failed
    #     #    if restarti == 1:
    #     #    # Restart FGM if not a descent direction
    #     #        restarti = 0
    #     #        alpha_next = alpha0
    #     #        Ys = S.copy()
    #     #        Yu = U.copy()
    #     #        Yb = B.copy()
    #     #        error_next = error
    #     #        #print(" No descent: Restart FGM")
    #
    #     #        # Reinitialize step length
    #     #        eS,_ = np.linalg.eig(S)
    #     #        L = (np.max(eS)/ np.min(eS))**2
    #     #        # Use information from the first step: how many steps to decrease
    #     #        step = (1.0/L)/(lsparam**inner_iter0)
    #     #    elif (restarti == 0): # no previous restart/descent direction
    #     #        error_next = error
    #     #        break
    #     #else:
    #         #print('dose updated')
    #     #restarti = 1
    #     #if (gradient == 1):
    #     #    beta = 0
    #     #Ys = Sn + beta * (Sn - S)
    #     #Yu = Un + beta * (Un - U)
    #     #Yb = Bn + beta * (Bn - B)
    #     # Keep new iterates in memory
    #     i = i + 1
    #     #error = error_next
    #     #alpha = alpha_next
    #     # Check if error is small (1e-6 relative error)
    Kd = np.linalg.inv(S).dot(U).dot(B).dot(S)
    return S, U, B, Kd, 0.
