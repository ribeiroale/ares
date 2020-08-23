from math import log
from math import gamma as gammafunction


def point_predictor(last_failure, failure_model):
    gamma, alpha, beta = failure_model[1]
    ret = (1 / alpha + last_failure ** beta) ** (1/beta)
    ret2 = ret - last_failure
    return ret, ret2


def interval_predictor(last_failure, failure_model, ε_1=.025, ε_2=.025):
    gamma, alpha, beta = failure_model[1]
    phi = (alpha * gammafunction(1+1/gamma)) ** gamma
    T_L = (last_failure ** beta + (1/phi * log(1 / (1-ε_1))) ** (1/gamma)) ** (1/beta)
    T_U = (last_failure ** beta + (1/phi * log(1 / (  ε_2))) ** (1/gamma)) ** (1/beta)
    return [T_L, T_U]
