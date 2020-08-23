from scipy.stats import uniform, expon, rayleigh, weibull_min, gamma, gengamma, invgamma, gompertz, lognorm, exponweib
from scipy.stats import kstest
from numpy import log, product


def downtime_accepted_models(D=list(), alpha=.05):
    params = list()
    params.append(uniform.fit(D))
    params.append(expon.fit(D))
    params.append(rayleigh.fit(D))
    params.append(weibull_min.fit(D))
    params.append(gamma.fit(D))
    params.append(gengamma.fit(D))
    params.append(invgamma.fit(D))
    params.append(gompertz.fit(D))
    params.append(lognorm.fit(D))
    params.append(exponweib.fit(D))

    llf_value = list()
    llf_value.append(log(product(uniform.pdf(D, *params[0]))))
    llf_value.append(log(product(expon.pdf(D, *params[1]))))
    llf_value.append(log(product(rayleigh.pdf(D, *params[2]))))
    llf_value.append(log(product(weibull_min.pdf(D, *params[3]))))
    llf_value.append(log(product(gamma.pdf(D, *params[4]))))
    llf_value.append(log(product(gengamma.pdf(D, *params[5]))))
    llf_value.append(log(product(invgamma.pdf(D, *params[6]))))
    llf_value.append(log(product(gompertz.pdf(D, *params[7]))))
    llf_value.append(log(product(lognorm.pdf(D, *params[8]))))
    llf_value.append(log(product(exponweib.pdf(D, *params[9]))))

    AIC = list()
    AIC.append(2 * len(params[0]) - 2 * llf_value[0])
    AIC.append(2 * len(params[1]) - 2 * llf_value[1])
    AIC.append(2 * len(params[2]) - 2 * llf_value[2])
    AIC.append(2 * len(params[3]) - 2 * llf_value[3])
    AIC.append(2 * len(params[4]) - 2 * llf_value[4])
    AIC.append(2 * len(params[5]) - 2 * llf_value[5])
    AIC.append(2 * len(params[6]) - 2 * llf_value[6])
    AIC.append(2 * len(params[7]) - 2 * llf_value[7])
    AIC.append(2 * len(params[8]) - 2 * llf_value[8])
    AIC.append(2 * len(params[9]) - 2 * llf_value[9])

    model = list()
    model.append(["uniform", params[0], kstest(D, "uniform", params[0])[1], AIC[0]])
    model.append(["expon", params[1], kstest(D, "expon", params[1])[1], AIC[1]])
    model.append(["rayleigh", params[2], kstest(D, "rayleigh", params[2])[1], AIC[2]])
    model.append(["weibull_min", params[3], kstest(D, "weibull_min", params[3])[1], AIC[3]])
    model.append(["gamma", params[4], kstest(D, "gamma", params[4])[1], AIC[4]])
    model.append(["gengamma", params[5], kstest(D, "gengamma", params[5])[1], AIC[5]])
    model.append(["invgamma", params[6], kstest(D, "invgamma", params[6])[1], AIC[6]])
    model.append(["gompertz", params[7], kstest(D, "gompertz", params[7])[1], AIC[7]])
    model.append(["lognorm", params[8], kstest(D, "lognorm", params[8])[1], AIC[8]])
    model.append(["exponweib", params[9], kstest(D, "exponweib", params[9])[1], AIC[9]])

    accepted_models = [i for i in model if i[2] > alpha]

    if accepted_models:
        aic_values = [i[3] for i in accepted_models]
        final_model = min(range(len(aic_values)), key=aic_values.__getitem__)
        return accepted_models, accepted_models[final_model]
    elif not accepted_models:
        aic_values = [i[3] for i in model]
        final_model = min(range(len(aic_values)), key=aic_values.__getitem__)
        return model, model[final_model]
