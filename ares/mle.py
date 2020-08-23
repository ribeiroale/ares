from numpy import log, zeros_like, exp, array
from math import gamma as gammafunction
from statsmodels.base.model import GenericLikelihoodModel


def loglike_WPLP(T, beta, gamma, phi):
    n = len(T) - 1
    ret = n * (log(phi) + log(beta) + log(gamma)) + \
          (beta - 1) * log(T[1]) + (gamma - 1) * log( T[1] ** beta) - phi * T[1] ** (beta * gamma) + \
          sum([(beta - 1) * log(T[i]) + (gamma - 1) * log(T[i] ** beta - T[i-1] ** beta) - \
          phi * (T[i] ** beta - T[i-1] ** beta) ** gamma for i in range(2, n+1)])
    return ret


def loglike_NHPP(T, beta, phi):
    return loglike_WPLP(T, beta=beta, phi=phi, gamma=1)


def loglike_HPP(T, phi):
    return loglike_NHPP(T, beta=1, phi=phi)


def loglike_WRP(T, gamma, phi):
    return loglike_WPLP(T, beta=1, phi=phi, gamma=gamma)


class WPLP_Model(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = zeros_like(endog)

        super(WPLP_Model, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        beta, gamma, phi = exp(params[0]), exp(params[1]), exp(params[2])
        return -loglike_WPLP(self.endog, beta=beta, gamma=gamma, phi=phi)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            start_params = array([-.5, -.5, -.5])

        return super(WPLP_Model, self).fit(start_params=start_params,
                                           maxiter=maxiter, maxfun=maxfun, **kwds)


class NHPP_Model(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = zeros_like(endog)

        super(NHPP_Model, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        beta, phi = exp(params[0]), exp(params[1])
        return -loglike_NHPP(self.endog, beta=beta, phi=phi)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            start_params = array([-.5, -.5])

        return super(NHPP_Model, self).fit(start_params=start_params,
                                           maxiter=maxiter, maxfun=maxfun, **kwds)


class HPP_Model(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = zeros_like(endog)

        super(HPP_Model, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        phi = exp(params[0])
        return -loglike_HPP(self.endog, phi=phi)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            start_params = array([-.5])

        return super(HPP_Model, self).fit(start_params=start_params,
                                          maxiter=maxiter, maxfun=maxfun, **kwds)


class WRP_Model(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = zeros_like(endog)

        super(WRP_Model, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        gamma, phi = exp(params[0]), exp(params[1])
        return -loglike_WRP(self.endog, phi=phi, gamma=gamma)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            start_params = array([-.5, -.5])

        return super(WRP_Model, self).fit(start_params=start_params,
                                          maxiter=maxiter, maxfun=maxfun, **kwds)


def alpha_value(gamma, phi):
    return phi ** (1/gamma) / (gammafunction(1+1/gamma))


def failure_models(T=list()):
    model_WPLP = WPLP_Model(T)
    result_WPLP = model_WPLP.fit()
    model_NHPP = NHPP_Model(T)
    result_NHPP = model_NHPP.fit()
    model_HPP = HPP_Model(T)
    result_HPP = model_HPP.fit()
    model_WRP = WRP_Model(T)
    result_WRP = model_WRP.fit()

    models_loglike_values = [result_WPLP.llf, result_NHPP.llf, result_HPP.llf, result_WRP.llf]
    models_aic_values = [result_WPLP.aic, result_NHPP.aic, result_HPP.aic, result_WRP.aic]
    models_bic_values = [result_WPLP.bic, result_NHPP.bic, result_HPP.bic, result_WRP.bic]

    beta_WPLP, gamma_WPLP, phi_WPLP = exp(result_WPLP.params)
    alpha_WPLP = alpha_value(gamma_WPLP, phi_WPLP)
    CI_WPLP = exp(result_WPLP.conf_int(alpha=.05))
    betaLower_WPLP, betaUpper_WPLP = CI_WPLP[0]
    gammaLower_WPLP, gammaUpper_WPLP = CI_WPLP[1]
    phiLower_WPLP, phiUpper_WPLP = CI_WPLP[2]
    alphaLower_WPLP = alpha_value(gammaLower_WPLP, phiLower_WPLP)
    alphaUpper_WPLP = alpha_value(gammaUpper_WPLP, phiUpper_WPLP)

    gamma_NHPP = 1
    beta_NHPP, phi_NHPP = exp(result_NHPP.params)
    alpha_NHPP = alpha_value(gamma_NHPP, phi_NHPP)
    CI_NHPP = exp(result_NHPP.conf_int(alpha=.05))
    betaLower_NHPP, betaUpper_NHPP = CI_NHPP[0]
    gammaLower_NHPP, gammaUpper_NHPP = 1, 1
    phiLower_NHPP, phiUpper_NHPP = CI_NHPP[1]
    alphaLower_NHPP = alpha_value(gammaLower_NHPP, phiLower_NHPP)
    alphaUpper_NHPP = alpha_value(gammaUpper_NHPP, phiUpper_NHPP)

    gamma_HPP = 1
    beta_HPP = 1
    phi_HPP = exp(result_HPP.params)[0]
    alpha_HPP = alpha_value(gamma_HPP, phi_HPP)
    CI_HPP = exp(result_HPP.conf_int(alpha=.05))
    betaLower_HPP, betaUpper_HPP = 1, 1
    gammaLower_HPP, gammaUpper_HPP = 1, 1
    phiLower_HPP, phiUpper_HPP = CI_HPP[0]
    alphaLower_HPP = alpha_value(gammaLower_HPP, phiLower_HPP)
    alphaUpper_HPP = alpha_value(gammaUpper_HPP, phiUpper_HPP)

    beta_WRP = 1
    gamma_WRP, phi_WRP = exp(result_WRP.params)
    alpha_WRP = alpha_value(gamma_WRP, phi_WRP)
    CI_WRP = exp(result_WRP.conf_int(alpha=.05))
    betaLower_WRP, betaUpper_WRP = 1, 1
    gammaLower_WRP, gammaUpper_WRP = CI_WRP[0]
    phiLower_WRP, phiUpper_WRP = CI_WRP[1]
    alphaLower_WRP = alpha_value(gammaLower_WRP, phiLower_WRP)
    alphaUpper_WRP = alpha_value(gammaUpper_WRP, phiUpper_WRP)

    model = list()
    model.append(["WPLP", [gamma_WPLP, alpha_WPLP, beta_WPLP],
                  [[gammaLower_WPLP, gammaUpper_WPLP], [alphaLower_WPLP, alphaUpper_WPLP],
                   [betaLower_WPLP, betaUpper_WPLP]], models_loglike_values[0], models_aic_values[0],
                  models_bic_values[0]])
    model.append(["NHPP", [gamma_NHPP, alpha_NHPP, beta_NHPP],
                  [[gammaLower_NHPP, gammaUpper_NHPP], [alphaLower_NHPP, alphaUpper_NHPP],
                   [betaLower_NHPP, betaUpper_NHPP]], models_loglike_values[1], models_aic_values[1],
                  models_bic_values[1]])
    model.append(["HPP", [gamma_HPP, alpha_HPP, beta_HPP],
                  [[gammaLower_HPP, gammaUpper_HPP], [alphaLower_HPP, alphaUpper_HPP], [betaLower_HPP, betaUpper_HPP]],
                  models_loglike_values[2], models_aic_values[2], models_bic_values[2]])
    model.append(["WRP", [gamma_WRP, alpha_WRP, beta_WRP],
                  [[gammaLower_WRP, gammaUpper_WRP], [alphaLower_WRP, alphaUpper_WRP], [betaLower_WRP, betaUpper_WRP]],
                  models_loglike_values[3], models_aic_values[3], models_bic_values[3]])

    return model
