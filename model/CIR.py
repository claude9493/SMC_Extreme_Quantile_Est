# -*- coding: utf-8 -*-
"""
# To-do
1. Filtering
2. Tail probability estimation
3. Prediction: analyze the probability of state being extreme in next time stage.

# Proposal
1. (Mar 04, 2021) I plan to save proposals inside the CIR class if possible and specify selected proposal by attribute
    CIR.props which is key of the dictionary Proposals which helps like the string proposal name to proposal density function.
    - Note: the modified CIR model, CIR_mod, changed the way of computing weights, we put it in an independent class
      temporaly. Perhaps someday I will ensemble it into the CIR class.
    @2021030401
2. (Mar 04, 2021) About multi-proposal, as the proposals are all in the CIR class now, we can define the multi-proposal
    easily for given proposals and combining weights.

# Outlier
1. As the CIR model with t proposal density are included in the CIR class after update @2021030401, the CIR_t class is
   now used to generate observations from t-distribution.

"""

import itertools
import random
import json
from collections import namedtuple
from functools import partial

import matplotlib.pyplot as plt
# %%
import numpy as np
import particles.distributions as dists
import seaborn as sns
from particles import state_space_models as ssm
from particles import collectors
from scipy import stats

from model.distributions import (ncx2, truncated_ncx2, t, t_nn, truncated_t, Mvt)
from loguru import logger

# %% CIR class
class CIR(ssm.StateSpaceModel):
    """Class for CIR model.

    Parameters
    ----------
    kappa          : mean reversion parameter
    theta          : long term mean the interest rate reverts to
    sigma          : volatility parameter
    lam (lambda)   : risk premium parameter
    tau            : maturity associated with the observed yields.
    Delta          : time interval
    H              : diagnol of the diagnoal variance–covariance matrix errors in observed equation for yields
    proposal       : a string variable indicating the proposal density

    ~~Here we assume that all observed interest rates $y_t$ have same time to maturities $\tau$. For details of model
    and parameters, please refer to S2W3 report or Rossi(2010).~~

    TO-DO:
        1. [ ] Make the degree of freedom of proposal_t editible. And use t distribution as proposal0. (Mar 04, 2021)
        2. [ ] Multi-proposal framework. (Mar 04, 2021)

    UPDATE 20210224:
        Now let's try to introduce multivariate observed yields with different maturities $\tau$ and variance $H_{ii}$.

    UPDATE 20200304:
        I decided to use a string indicator to specify proposal used in the algorithm. In order to reduce the number of
        classes. All available proposals are written at the end of CIR class and stored in dictionary `Proposals`. Use
        Proposals.get('name').(t, xp, data) to call them.


    Reference
    ----------
        De Rossi, G. (2010). Maximum likelihood estimation of the Cox–Ingersoll–Ross model using particle filters.Computational Economics, 36(1), 1-16.
    """
    default_params = {
        'kappa': 0.169,
        'theta': 6.56,
        'sigma': 0.321,
        'lam': -0.201
    }

    # Define proposals
    ''' Old proposal_normal (Rossi 2010)
    def proposal_normal(self, t, xp, data):
        a = self.a
        if self.num_yields == 1:
            # a = self.Btau ** 2 / (2 * self.H)
            # b = -(data[t] - self.Atau) * self.Btau / self.H
            b = -((data[t]+xp)/2 - self.Atau) * self.Btau / self.H
            # c = (data[t] - self.Atau)**2 / (2*self.H)
        else:
            # a = sum(self.Btau ** 2) / (2 * self.H)
            # b = -((data[t] - self.Atau) * self.Btau).sum() / self.H
            b = - (((data[t]/2 - self.Atau) * self.Btau).sum() + xp/2) / self.H
            # c = sum((data[t] - self.Atau)**2) / (2*self.H)
        return dists.TruncNormal(mu=-b / (2 * a), sigma= np.sqrt(1 / (2 * a)), a=0., b=np.inf)
    '''

    def opt_proposal_param(self, t, xp, data):
        '''
        Compute mean $\mu$ and standard deviation $\sigma$ of the optimal normal-type proposal.
        (taking normal approximation to p(x_{t}|x_{t-1}))
        '''
        mu_y = (self.Btau * (data[t] + self.Atau)).sum() / np.square(self.Btau).sum()
        mu_x = self.k / 2 / self.c + np.exp(-self.kappa*self.Delta)*xp  # k = 2*q+2
        var_y = self.H / np.square(self.Btau).sum()
        var_x = self.k / 2 / self.c**2 + 2 * np.exp(-self.kappa*self.Delta) * xp / self.c
        params = {'mu': (mu_x*var_y+mu_y*var_x)/(var_x+var_y),
                  'sigma': np.sqrt(var_x*var_y/(var_x+var_y))}
        return params

    # Optimal proposal (taking normal approximation to p(x_{t}|x_{t-1}))
    def proposal_normal(self, t, xp, data):
        return dists.TruncNormal(a=0., b=np.inf, **self.opt_proposal_param(t, xp, data))

    def proposal_normal_tail(self, t, xp, data, lower_tail=False):
        '''
        generate particles in the tail of normal proposal
        '''
        q = self.ptq
        params = self.opt_proposal_param(t, xp, data)
        if lower_tail:
            ta = 0
            # For the scipy.stats.truncnorm, a and b are defined over the domain of the standard normal, we need to
            # convert clip values for a specific mean and standard deviation.
            # my_a = 0, a_standard = (my_a-mu)/sigma
            tb = stats.truncnorm(a=(0.-params['mu'])/params['sigma'], b=np.inf, loc=params['mu'],
                                 scale=params['sigma']).ppf(1-q)
        else:
            ta = stats.truncnorm(a=(0.-params['mu'])/params['sigma'], b=np.inf, loc=params['mu'],
                                 scale=params['sigma']).ppf(q)
            tb = np.inf
        # a = (ta - params['mu']) / params['sigma']
        # b = (tb - params['mu']) / params['sigma']
        # return dists.TruncNormal(a=a, b=b, **params)
        # But the dists.TruncNormal, it transform the input a and b to standard normal domain for us before calling
        # the scipy.stats.truncnorm method.
        return dists.TruncNormal(a=ta, b=tb, **params)

    ''' Old proposal_normal (Rossi 2010)
    def proposal_t(self, t, xp, data):
        a = self.a
        if self.num_yields == 1:
            # a = self.Btau ** 2 / (2 * self.H)
            # b = -(data[t] - self.Atau) * self.Btau / self.H
            b = -((data[t] + xp) / 2 - self.Atau) * self.Btau / self.H
            # c = (data[t] - self.Atau)**2 / (2*self.H)
        else:
            # a = sum(self.Btau ** 2) / (2 * self.H)
            # b = -((data[t] - self.Atau) * self.Btau).sum() / self.H
            b = - (((data[t] / 2 - self.Atau) * self.Btau).sum() + xp / 2) / self.H
            # c = sum((data[t] - self.Atau)**2) / (2*self.H)
        params = {'df':self.tdf, 'loc':-b / (2 * a), 'scale':np.sqrt(1 / (2 * a))}
        return t_nn(**params)
    '''

    def proposal_t(self, t, xp, data):
        params = self.opt_proposal_param(t, xp, data)
        return t_nn(df=self.tdf, loc=params['mu'], scale=params['sigma'])

    '''
    def proposal_t_tail(self, t, xp, data, q=0.5, lower_tail=False):
        # Generate particles around the quantiles of non-negative t-proposal.
        a = self.a
        if self.num_yields == 1:
            # a = self.Btau ** 2 / (2 * self.H)
            b = -(data[t] - self.Atau) * self.Btau / self.H
            # c = (data[t] - self.Atau)**2 / (2*self.H)
        else:
            # a = sum(self.Btau ** 2) / (2 * self.H)
            # b = -((data[t] - self.Atau) * self.Btau).sum() / self.H
            b = - (((data[t] / 2 - self.Atau) * self.Btau).sum() + xp / 2) / self.H
            # c = sum((data[t] - self.Atau)**2) / (2*self.H)
        params = {'df':self.tdf, 'loc':-b / (2 * a), 'scale':np.sqrt(1 / (2 * a))}
        nc = 1 - stats.t.cdf(0, **params)
        if lower_tail:
            ta = 0
            tb = stats.t.ppf(q * nc + 1 - nc, **params)
            # tb = self.proposal_t(t, xp, data).ppf(q)  # q-th quantile of non-negative t distribution
        else:
            ta = stats.t.ppf(q * nc + 1 - nc, **params)
            # ta = self.proposal_t(t, xp, data).ppf(q)
            tb = np.inf
        return truncated_t(a=ta, b=tb, **params)
    '''

    def proposal_t_tail(self, t, xp, data, lower_tail=False):
        q = self.ptq
        params = self.opt_proposal_param(t, xp, data)
        params = {'df':self.tdf, 'loc':params['mu'], 'scale':params['sigma']}
        nc = 1 - stats.t(**params).cdf(0)
        if lower_tail:
            ta = 0
            tb = stats.t(**params).ppf((1-q) * nc + 1 - nc)  # q-th quantile of non-negative t distribution
            # tb = self.proposal_t(t, xp, data).ppf(q)
        else:
            ta = max(stats.t(**params).ppf(q * nc + 1 - nc), 0)
            # ta = self.proposal_t(t, xp, data).ppf(q)
            tb = np.inf
        return truncated_t(a=ta, b=tb, **params)

    def proposal_boot(self, t, xp, data):
        return self.PX(t=t, xp=xp)

    # What if we make the bootstrap filter multi-proposal like, by adding the tail of PX as a proposals
    def proposal_boot_tail(self, t, xp, data, q=0.75, lower_tail=False):
        '''
        Generate particles around the quantiles of non-negative t-proposal.
        '''
        k = 2 * 2 * self.kappa * self.theta / self.sigma ** 2  # + 2
        c = 2 * self.kappa / (self.sigma ** 2 * (1 - np.exp(-self.kappa * self.Delta)))
        l = 2 * c * np.exp(-self.kappa * self.Delta) * xp
        params = {'df': k, 'nc': l, 'scale': 1/(2*c)}
        if lower_tail:
            ta = 0
            tb = stats.ncx2.ppf(q, **params)
            # tb = self.proposal_t(t, xp, data).ppf(q)  # q-th quantile of non-negative t distribution
        else:
            ta = stats.ncx2.ppf(q, **params)
            # ta = self.proposal_t(t, xp, data).ppf(q)
            tb = np.inf
        return truncated_ncx2(a=ta, b=tb, **params)

    def __init__(self,
                 Delta=1 / 12,
                 tau=1, H=1,
                 proposal='normal',
                 tdf=20,
                 ptq=0.9,
                 **kwargs):

        if isinstance(tau, (np.ndarray, list)):
            tau = np.array(tau)
            self.num_yields = len(tau)
        else:
            self.num_yields = 1

        self.Proposals = {
            'normal'    : self.proposal_normal,
            'normalq'   : self.proposal_normal_tail,
            'normalql'  : partial(self.proposal_normal_tail, lower_tail=True),
            't'         : self.proposal_t,
            'tq'        : self.proposal_t_tail,
            'tql'       : partial(self.proposal_t_tail, lower_tail=True),
            'boot'      : self.proposal_boot,
            'bootq'     : self.proposal_boot_tail
        }

        assert proposal in self.Proposals.keys(), f"Selected proposal is not in the list, please input proposal from: {self.Proposals.keys()}."
        self.props = proposal

        super().__init__()
        self.__dict__.update(kwargs)

        self.Delta = Delta
        self.tau = tau
        self.H = H
        self.tdf = tdf
        self.ptq = ptq  # proposal tail quantile

        # degree of freedom and scaling parameter in the transition density
        # Refer to the section 2 of (Rossi 2010)
        self.k = 2 * 2 * self.kappa * self.theta / self.sigma ** 2  # + 2  # k = 2q+2
        self.c = 2 * self.kappa / (self.sigma ** 2 * (1 - np.exp(-self.kappa * self.Delta)))

        gamma = np.sqrt((self.kappa + self.lam) ** 2 + 2 * self.sigma ** 2)

        Btau = 1 / self.tau * (2 * (np.exp(gamma * self.tau) - 1)) / (
                    2 * gamma + (self.kappa + self.lam + gamma) * (np.exp(gamma * self.tau) - 1))

        Atau = (2 * self.kappa * self.theta / self.sigma ** 2) / self.tau * np.log(
            (2 * gamma * np.exp(self.tau * (self.kappa + self.lam + gamma) / 2)) / (
                        2 * gamma + (self.kappa + self.lam + gamma) * (np.exp(gamma * self.tau) - 1)))

        self.gamma, self.Atau, self.Btau = gamma, Atau, Btau

        if self.num_yields == 1:
            self.a = self.Btau ** 2 / (2 * self.H)
        else:
            self.a = np.square(Btau).sum() / (2 * self.H)


    def __repr__(self, detailed=False):
        if detailed:
            return f"""CIR model
            Parameters: {self.default_params}
            Maturities: {self.tau}
            Use '{self.props}' as proposal density."""
        else:
            return f"CIR model with {self.num_yields} dimension observation.Use '{self.props}' as proposal."

    def PX0(self):
        # return dists.TruncNormal(mu=0, sigma=1, a=0, b=np.inf)
        a = 2 * self.kappa / self.sigma ** 2
        b = 2 * self.kappa * self.theta / self.sigma ** 2 + 1
        return dists.Gamma(a, b)

    def PX(self, t, xp):
        '''
        The exact transition density
        '''
        # the k and c are constant, make them class attributes.
        # k = 2 * 2 * self.kappa * self.theta / self.sigma ** 2  # + 2
        # c = 2 * self.kappa / (self.sigma ** 2 * (1 - np.exp(-self.kappa * self.Delta)))
        l = 2 * self.c * np.exp(-self.kappa * self.Delta) * xp
        return ncx2(df=self.k, nc=l, scale=1/(2*self.c))

    def PY(self, t, xp, x):
        # print(f"PY is called, shape of x {x}, {x.shape}")
        if self.num_yields == 1:
            mu = -self.Atau + self.Btau * x
            scale = np.sqrt(self.H)
            return dists.Normal(mu, scale)
        else:
            # In particles\core.py\SMC reweight_particles call the fk model's logG and therefore PY here is called by
            # the Bootstrap filter. In this process, x is inputed as a list, containing all particles in one time step.
            # So the normal broadcast mechanism doen not work as desire.

            mu = [-self.Atau + self.Btau * xi for xi in x]
            cov = np.eye(self.num_yields) * self.H
            # cov = np.diag(np.asanyarray(self.H))
            # cov = np.zeros((self.num_yields, self.num_yields))
            # cov[:self.num_yields].flat[0::self.num_yields+1] = self.H
            return dists.MvNormal(loc=mu, cov=cov)

    def proposal0(self, data):
        # A test on using proposal0 same as PX0
        # a = 2 * self.kappa / self.sigma ** 2
        # b = 2 * self.kappa * self.theta / self.sigma ** 2 + 1
        # return dists.Gamma(a, b)
        return self.PX0()
        # return dists.Normal(0, 1)

    def proposal(self, t, xp, data):
        return self.Proposals.get(self.props)(t, xp, data)

    def __proposals__(self):
        return(self.Proposals.keys())


class CIR_t(CIR):
    """
    Class for CIR model with (generalized) t distribution as observation equation.

    History: this class was firtly used for CIR model with t-distribution as proposal density;
    """
    def __init__(self, df=3, emotion=False, l=10, nc=3, T=99, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.l = l
        # self.nc = nc
        self.nc = 1 + np.log(1 + nc * self.tau/sum(self.tau))
        self.T = T
        if emotion:
            self.emotion_start_t = random.randint(0, T-l)
        else:
            self.emotion_start_t = -1

    def PY(self, t, xp, x):
        if self.num_yields == 1:
            loc = -self.Atau + self.Btau * x
            scale = self.H
            if self.emotion_start_t <= t < self.emotion_start_t+self.l:
                loc *= self.nc
            return t(df=self.df, loc=loc, scale=scale)
        else:
            loc = -self.Atau + self.Btau * x
            cov = np.eye(self.num_yields) * self.H
            if self.emotion_start_t <= t < self.emotion_start_t+self.l:
                loc *= self.nc
            return dists.MvNormal(loc=loc, cov=cov)
            # return Mvt(df=self.df, loc=loc, cov=cov)

    # def proposal0(self, data):
    #     return t(df=1, loc=0, scale=1)
    #
    # def proposal(self, t, xp, data):
    #     if self.num_yields == 1:
    #         a = self.Btau ** 2 / (2 * self.H)
    #         b = -(data[t] - self.Atau) * self.Btau / self.H
    #         # c = (data[t] - self.Atau)**2 / (2*self.H)
    #     else:
    #         a = sum(self.Btau ** 2) / (2 * self.H)
    #         b = -((data[t] - self.Atau) * self.Btau).sum() / self.H
    #         # c = sum((data[t] - self.Atau)**2) / (2*self.H)
    #     return t_nn(df=1, loc=-b / (2 * a), scale=1 / (2 * a))


class CIR_mod(CIR):
    """
    Class for modified CIR model.

    Reference:
        Neslihanoglu, S., & Date, P. (2019). A modified sequential Monte Carlo procedure for the efficient recursive estimation of extreme quantiles. Journal of Forecasting, 38(5), 390-399.
    """

    def __init__(self, s=5, **kwargs):
        super().__init__(**kwargs)
        self.s = s

    def logG(self, t, xp, x):
        w_orig = np.exp(super().logG())
        w_mod = np.log((np.exp(-self.s * w_orig) - 1) / (np.exp(-self.s) - 1))
        return w_mod


class CIR_config():
    def __init__(self, record_id):
        self.id = record_id
        self.template = {
            'RECORD_ID': self.id,
            'MODEL': {
                'PARAMS': {'kappa': None, 'theta': None, 'sigma': None, 'lam': None},
                'SNR': None,
                'OBS': {'dim': None, 'tau': [], 'H': None, 'delta': None}
            },
            'ALG': {},
            'FIG': {'real_x': None, 'obs': None, 'f_mse': None, 'q_mse': None}
        }
        pass

    def load(self):
        with open(f"./Records/CIR{self.id}/record{self.id}.json") as json_file:
            config = json.load(json_file)
        logger.info(f"Configuration of {self.id} record is loaded successfully.")
        return config

    def generate(self, CIR, **kwargs):
        self.template['MODEL']['PARAMS'].update({
            'kappa': CIR.kappa,
            'theta': CIR.theta,
            'sigma': CIR.sigma,
            'lam': CIR.lam
        })
        self.template['MODEL']['OBS'].update({
            'dim': len(CIR.tau),
            'tau': CIR.tau.tolist(),
            'H': CIR.H,
            'delta': CIR.Delta
        })
        self.template['MODEL']['SNR'] =  CIR.sigma**2 * CIR.theta / CIR.H

        with open(f"./Records/CIR{self.id}/Record{self.id}.json", 'a') as json_file:
            json.dump(self.template, json_file, indent=4)
        logger.info(f"Configuration of {self.id} record is generated successfully.")

    def update(self):
        pass

def post_mean(W, x):
    '''
    Moment function used to estimate posterior mean of latent states.
    usage
    =====
        from CIR import post_mean
        alg = particles.SMC(fk=fk_boot, N=100, moments=post_mean)
    Note
    ====
    There is a moments collector defined as the subclass of Collector class, it uses the default moments function defined
    in fk class and the resampling class, which computes both filtering mean and variance. We more like the variance of
    the filtering mean estimator.
    '''
    return np.average(x, weights=W, axis=0)

def CIR_plot(real_x):
    """
    Plot the real states time series and the distribution.

    Args:
        real_x (numpy array): real states
    """
    real_x = np.asanyarray(real_x)
    T = len(real_x)
    fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw={"width_ratios": [1, 8], "wspace": 0}, figsize=(9, 6))
    ax[1].plot(np.linspace(0, T, T), real_x)
    sns.rugplot(real_x, axis='y', height=.1, lw=1, alpha=.8, ax=ax[0])
    sns.kdeplot(data=list(itertools.chain.from_iterable(real_x)), vertical=True, ax=ax[0])
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


# %%
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
