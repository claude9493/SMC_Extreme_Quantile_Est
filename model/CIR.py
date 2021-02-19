# -*- coding: utf-8 -*-
"""
# To-do
1. Filtering
2. Tail probability estimation
3. Prediction: analyze the probability of state being extreme in next time stage.

# Proposal

"""

#%%
import numpy as np
import particles
from particles import state_space_models as ssm
import particles.distributions as dists
import statsmodels
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from distributions import (ncx2, t, t_nn)
#%% CIR class
class CIR(ssm.StateSpaceModel):
    """
    Class for CIR model.

    Here we assume that all observed interest rates $y_t$ have same time to maturities $\tau$. For details of model and parameters, please refer to S2W3 report or Rossi(2010).
    
    Reference:
        De Rossi, G. (2010). Maximum likelihood estimation of the Cox–Ingersoll–Ross model using particle filters.Computational Economics, 36(1), 1-16.
    """
    default_params = {
        'kappa':0.169, 
        'theta':6.56, 
        'sigma':0.321, 
        'lam':-0.201
    }


    def __init__(self, Delta=1/12, H=1):
        super().__init__()
        tau = 1
        self.Delta = Delta
        self.H = H
        gamma = np.sqrt((self.kappa+self.lam)**2 + 2*self.sigma**2)
        Btau = 1/tau * (2*(np.exp(gamma*tau)-1)) / (2*gamma+(self.kappa+self.lam+gamma)*(np.exp(gamma*tau)-1))
        Atau = (2*self.kappa*self.theta/self.sigma**2 + 1) / tau * np.log((2*gamma*np.exp(tau*(self.kappa+self.lam+gamma)/2)) / (2*gamma+(self.kappa+self.lam+gamma)*(np.exp(gamma*tau)-1)))
        self.gamma, self.Atau, self.Btau = gamma, Atau, Btau


    def PX0(self):
        # return dists.TruncNormal(mu=0, sigma=1, a=0, b=np.inf)
        a = 2*self.kappa / self.sigma**2
        b = 2*self.kappa*self.theta / self.sigma**2 + 1
        return dists.Gamma(a, b)

    def PX(self, t, xp):
        k = 2*2*self.kappa*self.theta / self.sigma**2 + 2
        c = 2*self.kappa/(self.sigma**2 * (1-np.exp(-self.kappa*self.Delta)))
        l = 2*c*np.exp(-self.kappa*self.Delta)*xp
        return ncx2(k=k, l=l, scale=2*c)

    def PY(self, t, xp, x):
        mu = -self.Atau + self.Btau*x
        sigma = self.H
        return dists.Normal(mu, sigma)

    def proposal0(self, data):
        return dists.Normal(0, 1)
    
    def proposal(self, t, xp, data):
        a = self.Btau**2/(2*self.H)
        b = -(data[t] - self.Atau)*self.Btau / self.H
        c = (data[t] - self.Atau)**2 / (2*self.H)
        return dists.TruncNormal(mu = -b/(2*a), sigma = 1/(2*a), a=0., b=np.inf)


class CIR_t(CIR):
    """
    Class for CIR model with (generalized) t distribution as proposal density.

    TO-DO:
    1. truncated t-distribution
    2. multi-proposal CIR
    """

    def proposal0(self, data):
        return t(df=1, loc=0, scale=1)

    def proposal(self, t, xp, data):
        a = self.Btau**2/(2*self.H)
        b = -(data[t] - self.Atau)*self.Btau / self.H
        c = (data[t] - self.Atau)**2 / (2*self.H)
        return t_nn(df=1, loc=-b/(2*a), scale = 1/(2*a))

class CIR_mod(CIR):
    """
    Class for modified CIR model.

    Reference:
        Neslihanoglu, S., & Date, P. (2019). A modified sequential Monte Carlo procedure for the efficient recursive estimation of extreme quantiles. Journal of Forecasting, 38(5), 390-399.
    """
    def __init__(self, s=5):
        super().__init__()
        self.s = s

    def logG(self, t, xp, x):
        w_orig = np.exp(self.super().logG())
        w_mod = np.log((np.exp(-self.s*w_orig)-1) / (np.exp(-self.s)-1))
        return w_mod


def CIR_plot(real_x):
    """
    Plot the real states time series and the distribution.

    Args:
        real_x (numpy array): real states
    """

    T = len(real_x)
    fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw={"width_ratios" : [1, 8], "wspace" : 0}, figsize=(9, 6))
    ax[1].plot(np.linspace(0,T, T), real_x)
    sns.rugplot(real_x, axis='y', height=.1, lw=1, alpha=.8, ax=ax[0]   )
    sns.kdeplot(data=list(itertools.chain.from_iterable(real_x)),   vertical=True, ax=ax[0])
    # Remove unnecessary frames, ticks, and labels
    ax[0].spines['right'].set_visible(False)
    ax[0].tick_params(bottom=False)
    ax[0].tick_params(labelbottom=False)
    ax[1].spines['left'].set_visible(False)
    ax[1].tick_params(left=False)
    # Axis label
    fig.text(0.5, 0.03, "time index", ha="center")
    ax[0].set_ylabel("interest rate")
    fig.text(0.5, 0.9, "True states and the density", ha="center")

#%%
if __name__ == "__main__":
    # %% Generate data from the CIR model
    T = 100
    my_CIR = CIR()
    real_x, real_y = my_CIR.simulate(T)

    # %% Plot the simulated data
    plt.plot(real_x)
    plt.xlabel("time index")
    plt.ylabel("interest rate")

# %%
