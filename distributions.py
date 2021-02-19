# -*- coding: utf-8 -*-
import numpy as np
import particles.distributions as dists
from scipy import stats

class ncx2(dists.ProbDist):
    """
    Class for noncentral chi-square distribution.

    Reference:
    Document of ncx2 in SciPy:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html
    
    WikiPedia page of Noncentral chi-squared distribution:
    https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
    """

    def __init__(self, k=1., l=1., scale=1):
        self.k = k
        self.l = l
        self.scale = scale

    def rvs(self, size=None):
        return stats.ncx2.rvs(df=self.k, nc=self.l, size=size) / self.scale

    def logpdf(self, x):
        return stats.ncx2.logpdf(x*self.scale, self.k, self.l)

    def ppf(self, x):
        return stats.ncx2.ppf(x*self.scale, self.k, self.l)

class t(dists.ProbDist):
    """
    Class for t distribution

    Reference:
    Document of t in SciPy:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    """
    def __init__(self, df=1, loc=0, scale=1):
        self.df = df
        self.loc = loc
        self.scale = scale

    def rvs(self, size=None):
        return stats.t.rvs(self.df, loc=self.loc, scale=self.scale, size=size)

    def logpdf(self, x):
        return stats.t.logpdf(x, self.df, loc=self.loc, scale=self.scale)

    def ppf(self, x):
        return stats.t.ppf(x, self.df, loc=self.loc, scale=self.scale)

class t_nn(t):
    def __init__(self, df=1, loc=0, scale=1):
        super().__init__(df, loc, scale)
        # Normalizing constant: area of the part smaller than 0 of the original t distribution.
        self.nc = stats.t.cdf(0, self.df, loc=self.loc, scale=self.scale)

    def rv(self):
        rv = super().rvs(size=1)
        return rv if rv >= 0 else self.rv()

    def rvs(self, size):
        return np.array([self.rv()[0] for _ in range(size)])
        # return self.loc + np.abs(stats.t.rvs(size, df=self.df)) * self.scale

    def logpdf(self, x):
        return np.log(np.exp(super().logpdf(x)) / (1-self.nc))

    def ppf(self, q):
        return super().ppf(q / (1-self.nc) + self.nc)
