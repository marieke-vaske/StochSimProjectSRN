import scipy.stats as stats
import numpy as np


def aCalc(x, c):
    a1 = c[0]*(x[0]*x[1])*(x[0]>=1 and x[1]>=1)
    a2 = c[1]*(x[2])*(x[2]>=1)
    a3 = c[2]*(x[2])*(x[2]>=1)
    return np.array([a1, a2, a3])
#%%
def update_J(J, y, c, tau):
    y0, y1 = y[0], y[1]
    c0, c1, c2 = c

    J[0,0] = 1 + tau*c0*y1
    J[0,1] = tau*c0*y0

    J[1,0] = tau*c0*y1
    J[1,1] = 1 + tau*c0*y0

    J[2,0] = -tau*c0*y1
    J[2,1] = -tau*c0*y0

    J[0,2] = -tau*c1
    J[1,2] = -tau*(c1+c2)
    J[2,2] = 1 + tau*(c1+c2)
    J[3,2] = -tau*c2

    # entries that are constant
    J[0,3] = J[1,3] = J[2,3] = 0
    J[3,0] = J[3,1] = 0
    J[3,3] = 1


def newtons(c, X, tau, tol, nu, Nmax=100):
    y = X.copy()
    a = aCalc(y, c)
    F = y - X - tau * (a @ nu)

    J = np.empty((4, 4))  # allocated once
    N = 0
    while np.linalg.norm(F) > tol and N < Nmax:
        update_J(J, y, c, tau)

        delY = np.linalg.solve(J, -F)
        y += delY

        a = aCalc(y, c)
        F = y - X - tau * (a @ nu)
        N += 1

    return y

def MC_specificSampleSize(solver, args, N, alpha=0.05):
    cAlph = stats.norm.ppf(1 - alpha / 2)
    X4 = np.zeros(N)

    for i in range(N):
        t, X = solver(*args)
        X4[i] = X[-1, 3]

    muN = 1 / N * sum(X4)
    sigN = np.sqrt(1 / (N - 1) * sum((X4 - muN) ** 2))

    halfwidth = sigN * cAlph / np.sqrt(N)

    return muN, halfwidth


def pilot_run(solver, args, Nb):
    X4 = np.zeros(Nb)

    for i in range(Nb):
        t, X = solver(*args)
        X4[i] = X[-1, 3]

    muN = 1 / Nb * sum(X4)
    sigN = np.sqrt(1 / (Nb - 1) * sum((X4 - muN) ** 2))

    return muN, sigN


def iterative_MC(solver, args, Nb=200, tol=0.2, Nmax=2000, alpha=0.05):
    cAlph = stats.norm.ppf(1 - alpha / 2)
    N = Nb

    muN, sigN = pilot_run(solver, args, Nb)

    while (sigN * cAlph / np.sqrt(N) > tol and N < Nmax):
        N = N + 1

        t, X = solver(*args)

        muN = (N / (N + 1)) * muN + 1 / (N + 1) * X[-1, 3]
        sigN = np.sqrt((N - 1) / N * sigN ** 2 + 1 / (N + 1) * (X[-1, 3] - muN) ** 2)

    halfwidth = sigN * cAlph / np.sqrt(N)

    return muN, halfwidth, N


def SSA_alg(c, X0, T, nu):

    '''
    SSA_alg takes the rate constants list c, inital condition array X0 and final time T.
    It returns an array of states X, and an array of times at wich a reaction occured.
    '''

    X = [X0]

    t = [0.0]
    n = 0
    while (t[-1]<T):
        a = aCalc(X[n], c)
        aSum = sum(a)

        u = np.random.uniform(0,1)
        P = a/aSum # Vector holding the probability of each reaction
        #F = np.array([P[0], P[0]+P[1] , P[0]+P[1]+P[2]])
        if u < P[0]:
            j = 0
        elif u < P[0]+P[1] :
            j = 1
        else :
            j = 2

        X.append(X[n]+nu[j])
        tau = np.random.exponential(1/aSum)

        t.append(t[-1] + tau)
        n += 1
    return np.array(t), np.array(X)

def implicit_tau_leaping(c, X0, T, tau, nu, tol=1e-10):
    t = np.arange(0, T, tau)
    X = np.zeros((len(t), 4))
    X[0] = X0

    for i in range(len(t)-1):
        y = newtons(c, X[i], tau, tol, nu)
        a = aCalc(y, c)
        P = np.random.poisson(a * tau)
        X[i+1] = X[i] + P @ nu

    return t, X

def explicit_tau_leaping(c, X0, T, tau, nu):
    t = np.arange(0, T, tau)
    X = np.zeros((len(t), 4))

    X[0] = X0

    for i in range(len(t) - 1):
        a = aCalc(X[i], c)
        P = np.random.poisson(lam=(a * tau))  # Shape (3,)
        X[i + 1] = X[i] + P @ nu
    return t, X