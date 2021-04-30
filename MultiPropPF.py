'''
Multiple Proposal Particle Filter module.

Overview
========

This model defines the following objects:

* `MultiPropFK`: the base infrastructure class for multi-proposal Feynman-Kac models
* `MultiPropPF`: the base class for multi-proposal particle filters

The Multi-proposal Feynman-Kac class
====================================

Based on the Guided Particle Filter class, few functions are modified to satisfy multiple proposals case. To define it
in particluar, one should:

    (a) satisfy all needs of the `GuidedPF` class, provide the state space model and observation data
    (b) specify the proposals and corresponding proportions by a dictionary

The Multi-proposal Particle Filter class
========================================

Based on the SMC class, an indicator set is added to store the category of particles in one step. And a few modifications
for linking the indicator set with Feynman Kac model are made. Usage of this class is same as the SMC class.

Example::

    from MultiPropPF import MultiPropFK, MultiPropPF
    fk = MultiPropFK(ssm = CIR(),  data=real_y, proposals={'t':0.5, 'boot':0.5 })
    pf = MultiPropPF(fk = fk, N=100, ESSrmin=1,resampling="multinomial")
    pf.run()

Multi-proposal input structure:
{
    't': 0.5,
    'boot': 0.5
}
means two proposal densities, t and bootstrap(PX) are used with mixture proportion (0.5, 0.5).
'''


import numpy as np
from numba import jit, njit
from particles import SMC
from particles.state_space_models import GuidedPF
from particles import resampling as rs


class MultiPropFK(GuidedPF):
    '''Guided filter with multiple proposals for a given state-space model

    Parameters
    ----------
    proposals: dict
        the proposals and mixture proportion, proposals should be implemented in the ssm class
    ssm: StateSpaceModel object
        the considered state-space model
    data: list-like
        the data

    Note
    ----
    Argument ssm must implement methods `proposal0` and `proposal`. All proposals specified in proposals must exist in
    the `Proposal` dictionary of the ssm.
    '''
    def __init__(self, ssm, data, proposals):
        super().__init__(ssm, data)
        self.proposals = proposals

    def M0(self, N):
        return self.ssm.proposal0(self.data).rvs(size=N)

    def M(self, t, xp, Ip):
        # distribution.rvs() function returns an array even when generating one particles, need to use concatenate method.
        return np.concatenate([self.ssm.Proposals.get(list(self.proposals)[ip])(t, x_p, self.data).rvs(size=1)
                               for ip, x_p in zip(Ip, xp)])
        # return self.ssm.proposal(t, xp, self.data).rvs(size=xp.shape[0])

    def logG(self, t, xp, x, Ip):
        if t == 0:
            return (self.ssm.PX0().logpdf(x)
                    + self.ssm.PY(0, xp, x).logpdf(self.data[0])
                    - self.ssm.proposal0(self.data).logpdf(x))
        else:
            part1 = self.ssm.PX(t, xp).logpdf(x) + self.ssm.PY(t, xp, x).logpdf(self.data[t])
            # distribution.logpdf() function returns a single value when one value is given, use array is enough.
            # Use $q_{I_p}$ in the denumerator. (old version)  ========================================================
            # part2 = np.array([self.ssm.Proposals.get(list(self.proposals)[ip])(t, xpi, self.data).logpdf(xi)
            #          for ip, xpi, xi in zip(Ip, xp, x)])

            # Use the $q_{\alpha}$ in the denumerator. ================================================================
            # $q_{\alpha} = \sum_{k=1}^{p}\alpha_{k}q_{k}
            part2 = np.array([[np.exp(self.ssm.Proposals.get(prop)(t, xpi, self.data).logpdf(xi)) * alpha for xpi, xi in zip(xp, x)]
                              for prop, alpha in self.proposals.items()]).sum(axis=0)
            return part1 - np.log(part2)


    # def Gamma(self, t, xp, u):
    #     return np.array([self.ssm.Proposals.get(list(self.proposals)[ip])(t, xpi, self.data).ppf(u)
    #                      for ip, xpi in zip(self.Ip, xp)])
    #     # return self.ssm.proposal(t, xp, self.data).ppf(u)



class MultiPropPF(SMC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.proposals = proposals
        self.Np = np.array(list(self.fk.proposals.values()))[:-1]*self.N
        self.Np = self.Np.astype(int)
        self.Np = np.append(self.Np, self.N-sum(self.Np))

        # self.Ip = np.repeat(0, self.N)  # indicator set for proposal
        # self.Ip = np.concatenate([np.repeat(i, int(a * self.N)) for i, a in enumerate(self.fk.proposals.values())])
        self.Ip = np.concatenate([np.repeat(i, a) for i, a in enumerate(self.Np)])
        np.random.shuffle(self.Ip)

    def renewIp(self):
        '''Renew the Ip indicator list
        How to call it from the SMC class after resampling
        '''
        # self.Ip = np.concatenate([np.repeat(i, int(a*self.N)) for i, a in enumerate(self.fk.proposals.values())])
        self.Ip = np.concatenate([np.repeat(i, a) for i, a in enumerate(self.Np)])
        np.random.shuffle(self.Ip)

    def reweight_particles(self):
        wgts_delta = self.fk.logG(self.t, self.Xp, self.X, self.Ip)
        # wgts_delta = np.array([self.fk.logG(self.t, xpi, xi, ip) for xpi, xi, ip in zip(self,Xp, self.X, self.Ip)])
        self.wgts = self.wgts.add(wgts_delta)
        # self.wgts = self.wgts.add(self.fk.logG(self.t, self.Xp, self.X))

    def resample_move(self):
        self.rs_flag = self.aux.ESS < self.N * self.ESSrmin
        if self.rs_flag:  # if resampling
            self.A = rs.resampling(self.resampling, self.aux.W)
            self.renewIp()  # Renew the indicator set after resampling
            self.Xp = self.X[self.A]
            self.reset_weights()
            self.X = self.fk.M(self.t, self.Xp, self.Ip)
        elif not self.fk.mutate_only_after_resampling:
            self.A = np.arange(self.N)
            self.Xp = self.X
            self.X = self.fk.M(self.t, self.Xp, self.Ip)

    def __next__(self):
        """One step of a particle filter.
        """
        if self.fk.done(self):
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.setup_auxiliary_weights()  # APF
            # The renew indicator set operation is moved to the resample_move function
            if self.qmc:
                self.resample_move_qmc()
            else:
                self.resample_move()
        self.reweight_particles()
        self.compute_summaries()
        self.t += 1

