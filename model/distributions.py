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

    def __init__(self, df=1., nc=1., scale=1):
        self.df = df
        self.nc = nc
        self.scale = scale

    def rvs(self, size=None):
        return stats.ncx2.rvs(df=self.df, nc=self.nc, scale=self.scale, size=size) #/ self.scale

    def logpdf(self, x):
        return stats.ncx2.logpdf(x, self.df, self.nc, scale=self.scale)
        # return np.log(np.exp(stats.ncx2.logpdf(x * self.scale, self.k, self.l)) * self.scale)

    def ppf(self, x):
        return stats.ncx2.ppf(x, self.df, self.nc, scale=self.scale)
        # return stats.ncx2.ppf(x, self.k, self.l) / self.scale

    @property
    def stats(self):
        return stats.ncx2.stats(df=self.df, nc=self.nc, scale=self.scale, moments='mv')


class truncated_ncx2(ncx2):
    def __init__(self, df=1., nc=1., scale=1, a=0, b=np.inf):
        super().__init__(df, nc, scale)
        self.a, self.b = float(min(a, b)), float(max(a, b))
        A = stats.ncx2.cdf(a, df=self.df, nc=self.nc, scale=self.scale)
        B = stats.ncx2.cdf(b, df=self.df, nc=self.nc, scale=self.scale)
        self.norm_const = B - A

    def rvs(self, size=None):
        samples = np.zeros((0,))
        while samples.shape[0] < size:
            s = super().rvs(size=size)
            accepted = s[(s >= self.a) & (s <= self.b)]
            samples = np.concatenate((samples, accepted), axis=0)
        samples = samples[:size]
        return samples

    def logpdf(self, x):
        return np.log(np.exp(super().logpdf(x)) / self.norm_const)

    def ppf(self, u):
        return super().ppf(q / self.norm_const + 1 - self.norm_const)

    @property
    def stats(self):
        return (stats.ncx2.stats(df=self.df, nc=self.nc, scale=self.scale, moments='mv'), q)


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

    @property
    def stats(self):
        return stats.t.stats(self.df, loc=self.loc, scale=self.scale, moments='mv')


class t_nn(t):
    '''
    t distribution nonnegative, implemented by a normalizing constant.
    '''

    def __init__(self, df=1, loc=0, scale=1):
        super().__init__(df, loc, scale)
        # Normalizing constant: area of the part smaller than 0 of the original t distribution.
        self.nc = 1 - stats.t.cdf(0, self.df, loc=self.loc, scale=self.scale)

    def rvs(self, size=None):
        samples = np.zeros((0,))
        while samples.shape[0] < size:
            s = super().rvs(size=size)
            accepted = s[s >= 0]
            samples = np.concatenate((samples, accepted), axis=0)
        samples = samples[:size]
        return samples

    def logpdf(self, x):
        return np.log(np.exp(super().logpdf(x)) / self.nc)

    def ppf(self, q):
        return super().ppf(q / self.nc + 1 - self.nc)

    @property
    def stats(self):
        return super().stats


class truncated_t(t):
    def __init__(self, df=1, loc=0, scale=1, a=0, b=np.inf):
        super().__init__(df, loc, scale)
        self.a, self.b = float(min(a, b)), float(max(a, b))
        A = stats.t.cdf(a, self.df, loc=self.loc, scale=self.scale)
        B = stats.t.cdf(b, self.df, loc=self.loc, scale=self.scale)
        self.nc = B - A

    def rvs(self, size=None):
        samples = np.zeros((0,))
        while samples.shape[0] < size:
            s = super().rvs(size=size)
            accepted = s[(s >= self.a) & (s <= self.b)]
            samples = np.concatenate((samples, accepted), axis=0)
        samples = samples[:size]
        return samples

    def logpdf(self, x):
        temp = np.log(np.exp(super().logpdf(x)) / self.nc)
        if isinstance(x, np.ndarray):
            temp[np.where((self.a > x) | (x > self.b))] = -np.inf
            # temp[(self.a > x) | (x > self.b)] = -np.inf
        else:
            if self.a > x or x > self.b:
                temp = -np.inf
        return temp

    def ppf(self, u):
        return super().ppf(q / self.nc + 1 - self.nc)


class Mvt(dists.ProbDist):
    def __init__(self, df, loc, cov):
        self.df = df
        self.loc = np.asarray(loc)
        self.cov = cov
        pass

    def rvs(self, size):
        # if self.df == np.inf:
        #     x = 1.
        # else:
        #     x = np.random.chisquare(self.df, size)/self.df
        # z = np.random.multivariate_normal(np.zeros(self.d), self.cov, (size,))
        # return self.loc + z/np.sqrt(x)[:,None]
        if isinstance(self.loc, list):
            assert size in self.loc.shape, "If loc is a list of vectors, it should have same length with size."
            return [stats.multivariate_t(df=self.df, loc=mu, shape=self.cov).rvs(size=1) for mu in self.loc]
        else:
            return stats.multivariate_t(df=self.df, loc=self.loc, shape=self.cov).rvs(size=size)

    def logpdf(self, x):
        return stats.multivariate_t(df=self.df, loc=self.loc, shape=self.cov).logpdf(x=x)

    def ppf(self, q):
        pass
