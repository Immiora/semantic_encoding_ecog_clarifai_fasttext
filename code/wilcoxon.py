from scipy import stats
from scipy.stats import distributions
import numpy as np
from numpy import (asarray, compress)
import warnings
from collections import namedtuple

# this is a modified version of scipy.stats.wilcoxon, the credit goes to the scipy team 

def wilcoxon(x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided"):
    """
    Calculate the Wilcoxon signed-rank test.
    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.
    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : {"pratt", "wilcox", "zsplit"}, optional. Default is "wilcox".
        "pratt":
            includes zero-differences in the ranking process,
            but drops the ranks of the zeros, see [4]_, (more conservative)
        "wilcox":
            discards all zero-differences, the default
        "zsplit":
            includes zero-differences in the ranking process and split the
            zero rank between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.
    alternative : {"two-sided", "greater", "less"}, optional
        The alternative hypothesis to be tested, see Notes. Default is
        "two-sided".
    Returns
    -------
    statistic : float
        If `alternative` is "two-sided", the sum of the ranks of the
        differences above or below zero, whichever is smaller.
        Otherwise the sum of the ranks of the differences above zero.
    pvalue : float
        The p-value for the test depending on `alternative`.
    See Also
    --------
    kruskal, mannwhitneyu
    Notes
    -----
    The test has been introduced in [4]_. Given n independent samples
    (xi, yi) from a bivariate distribution (i.e. paired samples),
    it computes the differences di = xi - yi. One assumption of the test
    is that the differences are symmetric, see [2]_.
    The two-sided test has the null hypothesis that the median of the
    differences is zero against the alternative that it is different from
    zero. The one-sided test has the null that the median is positive against
    the alternative that the it is negative (``alternative == 'less'``),
    or vice versa (``alternative == 'greater.'``).
    The test uses a normal approximation to derive the p-value. A typical rule
    is to require that n > 20. For smaller n, exact tables can be used to find
    critical values.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    .. [2] Conover, W.J., Practical Nonparametric Statistics, 1971.
    .. [3] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed
       Rank Procedures, Journal of the American Statistical Association,
       Vol. 54, 1959, pp. 655-667. :doi:`10.1080/01621459.1959.10501526`
    .. [4] Wilcoxon, F., Individual Comparisons by Ranking Methods,
       Biometrics Bulletin, Vol. 1, 1945, pp. 80-83. :doi:`10.2307/3001968`
    Examples
    --------
    In [4]_, the differences in height between cross- and self-fertilized
    corn plants is given as follows:
    >>> d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]
    Cross-fertilized plants appear to be be higher. To test the null
    hypothesis that there is no height difference, we can apply the
    two-sided test:
    >>> from scipy.stats import wilcoxon
    >>> w, p = wilcoxon(d)
    >>> w, p
    (24.0, 0.04088813291185591)
    Hence, we would reject the null hypothesis at a confidence level of 5%,
    concluding that there is a difference in height between the groups.
    To confirm that the median of the differences can be assumed to be
    positive, we use:
    >>> w, p = wilcoxon(d, alternative='greater')
    >>> w, p
    (96.0, 0.020444066455927955)
    This shows that the null hypothesis that the median is negative can be
    rejected at a confidence level of 5% in favor of the alternative that
    the median is greater than zero. The p-value based on the approximation
    is within the range of 0.019 and 0.054 given in [2]_.
    Note that the statistic changed to 96 in the one-sided case (the sum
    of ranks of positive differences) whereas it is 24 in the two-sided
    case (the minimum of sum of ranks above and below zero).
    """

    WilcoxonResult = namedtuple('WilcoxonResult', ('w_statistic', 'z_statistic', 'pvalue'))

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' "
                         "or 'pratt' or 'zsplit'")

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError("Alternative must be either 'two-sided', "
                         "'greater' or 'less'")

    if y is None:
        d = asarray(x)
    else:
        x, y = map(asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon. Aborting.')
        d = x - y

    if zero_method == "wilcox":
        # Keep all non-zero differences
        d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        warnings.warn("Sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    # return min for two-sided test, but r_plus for one-sided test
    # the literature is not consistent here
    # r_plus is more informative since r_plus + r_minus = count*(count+1)/2,
    # i.e. the sum of the ranks, so r_minus and the min can be inferred
    # (If alternative='pratt', r_plus + r_minus = count*(count+1)/2 - r_zero.)
    # [3] uses the r_plus for the one-sided test, keep min for two-sided test
    # to keep backwards compatability
    if alternative == "two-sided":
        T = min(r_plus, r_minus)
    elif alternative == "greater":
        T = r_plus
    else:
        T = r_minus
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = r[d != 0]

    replist, repnum = stats.find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = np.sqrt(se / 24)

    # apply continuity correction if applicable
    d = 0
    if correction:
        if alternative == "two-sided":
            d = 0.5 * np.sign(T - mn)
        elif alternative == "less":
            d = -0.5
        else:
            d = 0.5

    # compute statistic and p-value using normal approximation
    z = (T - mn - d) / se
    if alternative == "two-sided":
        prob = 2. * distributions.norm.sf(abs(z))
    elif alternative == "greater":
        # large T = r_plus indicates x is greater than y; i.e.
        # accept alternative in that case and return small p-value (sf)
        prob = distributions.norm.sf(z)
    else:
        prob = distributions.norm.cdf(z)

    return WilcoxonResult(T, z, prob)

# from scipy import stats
# import numpy as np
# from collections import namedtuple
#
# def wilcoxon(x, y=None, zero_method="wilcox", correction=False, alternative='two-sided'):
#     WilcoxonResult = namedtuple('WilcoxonResult', ('statistic', 'pvalue'))
#
#     if y is None:
#         d = np.asarray(x)
#     else:
#         x, y = map(np.asarray, (x, y))
#         if len(x) != len(y):
#             raise ValueError('Unequal N in wilcoxon.  Aborting.')
#         d = x - y
#
#     if zero_method == "wilcox": # Keep all non-zero differences
#         d = np.compress(np.not_equal(d, 0), d, axis=-1)
#
#     count = len(d)
#     r = stats.rankdata(abs(d))
#     r_plus = np.sum((d > 0) * r, axis=0)
#     r_minus = np.sum((d < 0) * r, axis=0)
#
#     if zero_method == "zsplit":
#         r_zero = np.sum((d == 0) * r, axis=0)
#         r_plus += r_zero / 2.
#         r_minus += r_zero / 2.
#
#     T = min(r_plus, r_minus)
#     mn = count * (count + 1.) * 0.25
#     se = count * (count + 1.) * (2. * count + 1.)
#
#     if zero_method == "pratt":
#         r = r[d != 0]
#
#     replist, repnum = stats.find_repeats(r)
#     if repnum.size != 0: # Correction for repeated elements.
#         se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()
#
#     se = np.sqrt(se / 24)
#     correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
#     z = (T - mn - correction) / se
#     prob = 2. * stats.distributions.norm.sf(abs(z))
#
#     if alternative == "two-sided":
#         return WilcoxonResult(T, prob)
#     elif alternative == "greater":
#         return WilcoxonResult(T, prob/2) if z > 0 else WilcoxonResult(T, 1 - prob/2)
#     elif alternative == "less":
#         return WilcoxonResult(T, prob/2) if z < 0 else WilcoxonResult(T, 1 - prob/2)
#     else:
#         raise ValueError("Alternative should be either 'two-sided' "
#                          "or 'less' or 'greater'")
