from scipy.stats import chi2


def GLRT(llf_Ha, llf_H0, r):
    LR = 2 * (llf_Ha - llf_H0)
    pvalue = chi2.sf(-LR, r)
    return pvalue


def loglikelihood_square(failure_models, alpha=.05):
    models_loglike_values = [failure_models[i][3] for i in range(0, len(failure_models))]

    # Compare HPP vs. NHPP
    p_value_trend = GLRT(models_loglike_values[2], models_loglike_values[1], 1)
    # Compare HPP vs WRP
    p_value_nonpoisson = GLRT(models_loglike_values[2], models_loglike_values[3], 1)

    if p_value_trend > alpha:
        # Compare NHPP vs TRP
        p_value = GLRT(models_loglike_values[1], models_loglike_values[0], 1)
        if p_value > alpha:
            final_model = failure_models[0]
            return final_model  # TRP
        else:
            final_model = failure_models[1]
            return final_model  # NHPP
    elif p_value_nonpoisson > alpha:
        # Compare WRP vs TRP
        p_value = GLRT(models_loglike_values[3], models_loglike_values[0], 1)
        if p_value > alpha:
            final_model = failure_models[0]
            return final_model  # TRP
        else:
            final_model = failure_models[3]
            return final_model  # WRP
    else:
        final_model = failure_models[2]
        return final_model  # HPP


def akaike_information_criterion(models):
    models_aic_values = [models[i][4] for i in range(0, len(models))]
    final_model = min(range(len(models_aic_values)), key=models_aic_values.__getitem__)
    return models[final_model]
