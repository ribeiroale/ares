from math import log
from math import gamma as gammafunction
from scipy.stats import uniform, expon, rayleigh, weibull_min, gamma, gengamma, invgamma, gompertz, lognorm, exponweib, t


def failure_repair_process(X=list(), D=list()):
    """
    Description: This function converts the time-between-failures (TBF) and the repair times to a single global time scale.
    Input  | X: list containing the TBF of the system.
           | D: list containing the downtime times between failures of the system.
    Output | T: failure and repair times in global time
           | S: state of the system at each event epoch
    """
    S, T = list(), list()
    N = max([len(X), len(D)])

    if X[0] == 0:
        X = X[1:]
    else:
        pass

    n = 0
    T.append(0)
    S.append(1)

    while n < N:
        for i in range(0, 2):
            if S[-1] == 1:
                S.append(S[-1] - 1)
                try:
                    T.append(T[-1] + X[n])
                except IndexError:
                    pass
            else:
                S.append(S[-1] + 1)
                try:
                    T.append(T[-1] + D[n])
                except IndexError:
                    pass
        n = n + 1
    return T, S


def next_ttr(final_downtime_model, CRN=None):
    dist = final_downtime_model[0]
    params = final_downtime_model[1]

    if dist == "uniform":
        return uniform.rvs(*params, random_state=CRN)
    elif dist == "expon":
        return expon.rvs(*params, random_state=CRN)
    elif dist == "rayleigh":
        return rayleigh.rvs(*params, random_state=CRN)
    elif dist == "weibull_min":
        return weibull_min.rvs(*params, random_state=CRN)
    elif dist == "gamma":
        return gamma.rvs(*params, random_state=CRN)
    elif dist == "gengamma":
        return gengamma.rvs(*params, random_state=CRN)
    elif dist == "invgamma":
        return invgamma.rvs(*params, random_state=CRN)
    elif dist == "gompertz":
        return gompertz.rvs(*params, random_state=CRN)
    elif dist == "lognorm":
        return lognorm.rvs(*params, random_state=CRN)
    elif dist == "exponweib":
        return exponweib.rvs(*params, random_state=CRN)


def next_ttf(previous_ttf, final_failure_model, CRN=None):
    gamma, alpha, beta = final_failure_model[1]

    U = uniform.rvs(random_state=CRN)
    tmpA = previous_ttf ** beta
    tmpB = 1 / (alpha * gammafunction(1 + 1 / gamma))
    tmpC = log(1 / (1 - U)) ** (1 / gamma)
    tmpD = (tmpA + tmpB * tmpC) ** (1 / beta)

    return tmpD


def sample_path(final_failure_model, final_downtime_model, timeHorizon, CRN=None):
    T, D, T_G, S = list(), list(), list(), list()

    T.append(0)
    D.append(0)
    S.append(0)
    T_G.append(0)

    n = 1

    while max(T_G) < timeHorizon:

        D.append(next_ttr(final_downtime_model, CRN))
        T.append(next_ttf(T[-1], final_failure_model, CRN))

        if n % 2 != 0:
            S.append(S[-1] + 1)
            T_G.append(T_G[0] + T[int((n + 1) / 2)] + sum(D[1:int((n - 1) / 2 + 1)]))
        else:
            S.append(S[-1] - 1)
            T_G.append(T_G[-1] + D[int(n / 2)])

        n = n + 1

    X = [0 if i == 0 else T[i] - T[i - 1] for i in range(0, len(T))]

    return T_G, X, D, S, T


def sim(k_max, final_failure_model, final_downtime_model, timeHorizon, numfail, CRN=None, iter_max=500):
    T_G_R, S_R = list(), list()
    X_R, D_R, T_R = list(), list(), list()

    while len(T_G_R) <= k_max:
        j = 0
        while j == 0 or len(T) < numfail + 1:
            T_G, X, D, S, T = sample_path(final_failure_model, final_downtime_model, timeHorizon=timeHorizon, CRN=CRN)
            if j == iter_max:
                break
            j = j + 1
        print(f"Tried {j} times")
        T_G_R.append(T_G)
        S_R.append(S)
        X_R.append(X)
        D_R.append(D)
        T_R.append(T)
    return T_G_R, S_R, X_R, D_R, T_R
