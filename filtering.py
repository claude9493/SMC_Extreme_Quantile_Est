# -*- coding: utf-8 -*-

'''
Created on 2021-01-09 00:11:18

In this file, we implement several filtering algorithms on the AR(1) 
model defined in model/AR1.py

Model:
    AR(1) model with known parameters
    $$ x_n = \rho x_{n-1} + N(0,\tau^2) $$
    $$ y_n = x_n + N(0,\sigma^2) $$
Aim:
    Estimate the unknown state at different time index, according to the observations
Algorithms:
    1. Bootstrap filter
    2. Particle filter
    3. LM-SMC
Visualization:
    Use waterfall plot to display the distribution of estimated states at selected time index
'''

#%% Load packages
import particles
from model.AR1 import (AR1, AR1_SMC, AR1_APF)
import numpy as np
from particles import state_space_models as ssm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

# %% Settings parameters
N = 1000  # 5000
T = 1 * 10 ** 2  # 5 * 10 ** 4
R = 100  # Number of independent replications
my_seed = 3035802483
# Use same random seeds in R independent replications
MAX_INT_32 = np.iinfo(np.int32).max
seeds = [np.random.randint(MAX_INT_32) for i in range(R)]

#%% Define model and generate real data
ar1 = AR1()
real_x, real_y = ar1.simulate(T)

def state_est(alg, real_x, name = "", **kwargs):
    """
    TODO:
    [v] Using kwargs to involve more plotting parameters, for providing a way to specify the xmax manually.
    2. Display the bias bewteen real states and the mode estimates or weighted average estimates of them, as a series plot verus the time index.

    Keywords arguments:
        xmin, xmax: range of states' values studied, default value is -5, 5

    Do the filtering step by step with the alg object, and draw the waterfall plot to illustrate the distribution of estimated states at selected time index.
    From the waterfall plot we may assess the accuracy of estimation.
    """
    T = len(alg.fk.data)

    fig = plt.figure(figsize=(10,5))
    ax = fig.gca(projection='3d')
    
    xmin, xmax = kwargs.get("xmin", -5), kwargs.get("xmax", 5)

    xs = np.arange(xmin, xmax, 0.1)
    verts = []
    ts = np.arange(0, T, 10)
    hrx = []

    for i in range(T):
        next(alg)
        if i%10 == 0:
            dens = sm.nonparametric.KDEUnivariate(alg.X)
            dens.fit()
            ds = dens.evaluate(xs)
            ds[0], ds[-1] = 0, 0
            verts.append(list(zip(xs, ds)))
            hrx.append(dens.evaluate(real_x[i]))
        
    poly = PolyCollection(verts,facecolor=(1,1,1,0.6))
    # poly.set_alpha(0.7)
    poly.set_edgecolor((0,0,0,1))
    ax.add_collection3d(poly, zs=ts, zdir='y')
    ax.scatter([real_x[i] for i in ts], ts, hrx)
    ax.set(title = f"Waterfall representation of filtering distributions {name}",
        xlabel = "State",
        ylabel = "Time index",
        zlabel = "Density",
        xlim3d = (xmin, xmax),
        ylim3d = (max(ts)+1,min(ts)-1),
        zlim3d = (0, 0.6),
    )    
    # ax.set_xlabel('State')
    # ax.set_xlim3d(xmin, xmax)
    # ax.set_ylabel('Time index')
    # ax.set_ylim3d(max(ts)+1,min(ts)-1)
    # ax.set_zlabel('Density')
    # ax.set_zlim3d(0, 0.6)
    # ax.set_title(f"Waterfall representation of filtering distributions {name}")

    ax.view_init(30, 60)

'''
#%% Bootstrap filter
fk_boot = ssm.Bootstrap(ssm=AR1(), data=real_y)
alg_boot = particles.SMC(fk=fk_boot, N=N, resampling="multinomial", summaries=True)
state_est(alg_boot, real_x, "(Bootstrap filter)")

# %% Particle filter
fk_PF = ssm.GuidedPF(ssm=AR1_SMC(), data=real_y)
alg_PF = particles.SMC(fk=fk_PF, N=N, ESSrmin=1, resampling='multinomial', store_history=True, compute_moments=False, online_smoothing=None, verbose=False)
state_est(alg_PF, real_x, "(Particle filter)")

# %% Auxiliary particle filter
ssm_APF = AR1_APF()
fk_APF = ssm.AuxiliaryPF(ssm = ssm_APF, data=real_y)
# fk_APF.logeta = fk_APF.ssm.logeta
fk_APF.logetat = np.zeros((N,))

alg_APF = particles.SMC(fk = fk_APF, N=N, ESSrmin=1, resampling='multinomial', store_history=True, compute_moments=False, online_smoothing=None, verbose=False)
print(fk_APF.isAPF)
state_est(alg_APF, real_x, "(Auxiliary particle filter)")

# %% independent particle filter
'''