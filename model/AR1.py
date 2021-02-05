# -*- coding: utf-8 -*-
import numpy as np
import particles
from particles import state_space_models as ssm
import particles.distributions as dists
import statsmodels
from scipy import stats

# TODO: define the AR(1) models frequently used
# TODO: define a function which generate AR(1) objects of specific type (values of parameter, number of known parameters)

class AR1(ssm.StateSpaceModel):
    # A basic AR(1) model with known parameters rho, tau and sigma, 
    # without specified proposals for SMC
    default_params = {'rho':.5, 'tau':1., 'sigma':1.}

    def PX0(self):
        return dists.Normal()

    def PX(self, t, xp):
        return dists.Normal(loc=self.rho * xp, scale=self.tau)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=self.sigma)

class AR1_SMC(AR1):
    # AR(1) model with best proposals
    def proposal0(self, data):
        loc = data[0]/(self.sigma**2 + 1)
        scale = np.sqrt(self.sigma**2 / (1+self.sigma**2))
        # return self.PX0()
        return dists.Normal(loc = loc, scale = scale)

    def proposal(self, t, xp, data):
        loc = (self.rho*xp*self.sigma**2 + data[t]*self.tau**2)/(self.tau**2 + self.sigma**2)
        scale = np.sqrt(self.tau**2 * self.sigma**2 / (self.tau**2 + self.sigma**2))
        return dists.Normal(loc= loc, scale=scale)


class AR1_APF(AR1_SMC):
    '''
    Information from the next observation is used to determine which particles should survive resampling at a given time.

    The APF is a look ahead method where at time n we try to predict which samples will be in regions of high probability masses at time n + 1.
    '''
    # def logeta(self, t, x, data):
    #     scale = np.sqrt(self.sigma**2 + self.tau**2)
    #     return dists.Normal(self.rho*x, scale).logpdf(data[t])

    def logeta(self, t, x, xp, data):
        "Log of auxiliary function at time t. The auxiliary function is $\frac{p(y_{t+1}|X_{t}^{i})}{p(y_{t}|x_{t-1}^{i})}$"
        scale = np.sqrt(self.sigma**2 + self.tau**2)
        try:
            return dists.Normal(self.rho*x, scale).logpdf(data[t+1]) - dists.Normal(self.rho*xp, scale).logpdf(data[t])
        except TypeError:
            return np.ones(x.shape)

     


class AR1_Ind(AR1_SMC):
    # def proposal0(self, data):
    #     loc = self.tau**2 / (self.sigma**2 + self.tau**2) * data[0]
    #     scale = np.sqrt(self.sigma**2 * self.tau**2 / (self.sigma**2 + self.tau**2))
    #     return dists.Normal(loc, scale)

    def proposal(self, t, xp, data):
        pass

class AR1_Twist(AR1_SMC):
    pass

class AR1_LMSMC(AR1):
    def proposal0(self, data):
        pass
    
    def proposal(self, t, xp, data):
        pass