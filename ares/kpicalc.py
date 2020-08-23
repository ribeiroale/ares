from math import gamma as gammafunction
from statistics import mean, stdev
from lifelines import KaplanMeierFitter
from numpy import exp, linspace, asarray, zeros_like, array, sqrt
from scipy.stats import t


t_student = t

def differentiate_scalar(res, mesh, n):
    """
    Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference.
    """
    x = mesh
    # x = numpy.linspace(a, b, n+1)  # mesh
    df = zeros_like(x)       # df/dx
    f_vec = res
    dx = x[1] - x[0]
    # Internal mesh points
    for i in range(1, n):
        df[i] = (f_vec[i+1] - f_vec[i-1])/(2*dx)
    # End points
    df[0]  = (f_vec[1]  - f_vec[0]) /dx
    df[-1] = (f_vec[-1] - f_vec[-2])/dx
    return df


def weibull_dist_func(x, failure_model):
    gamma, alpha, beta = failure_model[1]
    return 1 - exp(- (x * gammafunction(1+1/gamma)) ** gamma)


def cum_trend_func(x, failure_model):
    gamma, alpha, beta = failure_model[1]
    return alpha * x ** beta


def inv_cum_trend_func(x, failure_model):
    gamma, alpha, beta = failure_model[1]
    return (x / alpha) ** (1/beta)


def trend_func(x, failure_model):
    gamma, alpha, beta = failure_model[1]
    if x == 0:
        return float("inf")
    else:
        return alpha * beta * x ** (beta - 1)


def rs_method(n, T, failure_model):

    F = lambda x: weibull_dist_func(x, failure_model)
    trans_time_scale = cum_trend_func(asarray(T), failure_model)
    _t = max(trans_time_scale)
    d = _t / n
    t = linspace(0, _t, n + 1)

    M = list()
    M.append(0)

    F_1 = F(d)
    for i in range(1, len(t)):
        M.append((F(i * d) + sum([(M[j] - M[j - 1]) * F(d * (i - j + .5)) for j in range(1, i)]) - M[i - 1] * F_1) / (
                    1 - F_1))

    t2 = inv_cum_trend_func(t, failure_model)
    m_s = differentiate_scalar(M, t, n)
    trend = array([trend_func(i, failure_model) for i in t2])
    m = m_s * trend

    return t2, (M, m)


def intensity_function(t, failure_model, T=list(), ϵ=1e-8):
    """
    t    : time of interest.
    gamma: shape parameter of the Weibull distribution with expectation 1.
    alpha: shape parameter of the power law trend function.
    beta : scale parameter of the power law trend function.
    T    : time-to-failure data.
    """

    model = failure_model[0]
    gamma, alpha, beta = failure_model[1]

    if T[0] != 0:
        T = [0] + T
    else:
        pass

    if model == "WPLP":
        if t == 0:
            if beta < 1 or gamma < 1:
                return float("inf")
            else:
                0
        else:
            return (alpha * gammafunction(1 + 1 / gamma)) ** gamma * gamma * beta * t ** (beta - 1) * (
                        t ** beta - T[N(t - ϵ, T)] ** beta) ** (gamma - 1)
    elif model == "NHPP":
        return alpha * beta * t ** (beta - 1)
    elif model == "HPP":
        return alpha
    elif model == "WRP":
        return (alpha * gammafunction(1 + 1 / gamma)) ** gamma * gamma * t * (t - T[N(t - ϵ, T)]) ** (gamma - 1)


def N(t, T=list()):
    """
    t    : time of interest.
    T    : time-to-failure data.
    """
    if T[0] != 0:
        T = [0] + T
    else:
        pass
    if t <= 0:
        return 0
    else:
        return sum([1 if x <= t else 0 for x in T]) - 1


def Y(t, T=list(), S=list()):
    N = len(T)-1
    for n in range(0, N):
        if T[n] <= t and T[n+1] > t:
            ret = S[n]
        elif t >= T[n+1]:
            ret = S[n]
        else:
            pass
    return ret


def A(t, T=list(), S=list()):
    if t == 0:
        return 0
    else:
        r = len([i for i in T if i <= t]) - 1
        try:
            tmpA = (t - T[r]) * S[r+1]
        except IndexError:
            tmpA = 0
        return (sum([(T[i] - T[i-1]) * S[i] for i in range(1, r+1)]) + tmpA) / (t - T[0])


def A_Sim(t, T_G_R, S_R, alpha=0.1):
    """
    t    : time of interest.
    T_G_R: global time realisation of each replication.
    S_R  : state of the system at each event epoch for each replication.
    """
    K = len(T_G_R)
    df = K - 1
    if isinstance(t, list):
        mean_a = [mean([A(i, T_G_R[k], S_R[k]) for k in range(0, K)]) for i in t]
        std_a = [stdev([A(i, T_G_R[k], S_R[k]) for k in range(0, K)]) for i in t]
        conf_int = array(mean_a) * array(std_a) * abs(t_student.ppf(alpha/2, df)) / sqrt(K)
        return mean_a, conf_int
    else:
        return mean([A(t, T_G_R[k], S_R[k]) for k in range(0, K)])


def model_parameters(failure_model, disp=False):
    params = failure_model[1]
    ci_params = failure_model[2]
    gamma, alpha, beta = params

    if disp is False:
        return ((gamma, alpha, beta), ci_params)
    else:
        return print(f"α   = {alpha:.4f}\n" f"β   = {beta:.4f}\n" f"γ   = {gamma:.4f}\n")


def time_axis(intervals, T):
    t = list()
    for i in range(1, len(T)):
        sub = (T[i] - T[i-1]) / intervals
        t = t + [T[i-1]] + [T[i-1] + j * sub for j in range(1, intervals)]
    return t


def time_between_failures(N, rep_data):
    K = len(rep_data)
    TBF = list()
    for n in range(0, N+1):
        TBF.append([])
        k = 0
        while k < K:
            try:
                tmp = rep_data[k][n]
            except IndexError:
                break
            TBF[-1].append(tmp)
            k = k + 1
    return TBF


def reliability(n, N, rep_data):
    TBF = time_between_failures(N, rep_data)
    kmf = KaplanMeierFitter()
    kmf.fit(TBF[n], label=f'Kaplan Meier Estimate for the {n}th TBF')
    return kmf


def num_failures(T):
    x = linspace(0, max(T))
    y = [N(i, T) for i in x]
    return x, y
