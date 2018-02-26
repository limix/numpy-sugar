def normal_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


def normal_icdf(x):
    from scipy.stats import norm
    return -norm.isf(x)


__all__ = ["normal_cdf", "normal_icdf"]
