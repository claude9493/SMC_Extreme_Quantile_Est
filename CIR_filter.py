# -*- coding: utf-8 -*-
#%% Load packages
import datetime
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import particles
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from particles import state_space_models as ssm
from multiprocessing.pool import Pool
from multiprocessing import Process, Queue
from functools import partial

import evaluate
import filtering
from model.CIR import (CIR, CIR_mod, CIR_plot)

# %% Settings parameters
N = 100  # 5000
T = 1 * 10 ** 2  # 5 * 10 ** 4
R = 100  # Number of independent replications
my_seed = 3035802483
# Use same random seeds in R independent replications
MAX_INT_32 = np.iinfo(np.int32).max
seeds = [np.random.randint(MAX_INT_32) for i in range(R)]

#%% Define model and generate real data
cir = CIR()
real_x, real_y = cir.simulate(T)
fk_PF = ssm.GuidedPF(ssm=CIR(), data=real_y)

#%% Save the generated data
name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(f"./Records/CIR{name}")
np.savez(file=f"./Records/CIR{name}/data", real_x=real_x, real_y=real_y)
print(f"Data saved, CIR{name}")

#%% Load data from saved npz file
name = "20210204-225727"
temp = np.load(file=f"./Records/CIR{name}/data.npz")
real_x, real_y = temp.get("real_x"), temp.get("real_y")
fk_PF = ssm.GuidedPF(ssm=CIR(), data=real_y)

#%% Plot the real_x
CIR_plot(real_x)
plt.savefig(f"./Records/CIR{name}/real_states.png")

#%% Function for repeated filtering simulation
default_args = {"N":100, "ESSrmin":1, "resampling":"multinomial", "store_history":True, "compute_moments":False, "online_smoothing":None, "verbose":False}

def simulate_single(fk, alg_args=default_args, queue):
    t0 = time.time()
    alg = particles.SMC(fk=fk, **alg_args)
    alg.run()
    t1 = time.time()   
    state_est = list(map(lambda i: np.average(alg.hist.X[i], weights=alg.hist.wgts[i].W), range(T)))
    queue.put((state_est, t1-t0))
    # return state_est, t1-t0

def repeated_simulation(fk, R=R, alg_args = default_args):
    num_cores = multiprocessing.cpu_count()
    f = partial(simulate_single, fk, alg_args)
    res = Parallel(n_jobs=num_cores)(delayed(f)(i) for i in range(R))

    state_ests = res[:,0]
    running_time = res[:,1]

    average_state_est = np.mean(state_ests, axis=0)
    return average_state_est, np.mean(running_time)


#%% Particle filter SMC_10K
alg_PF = particles.SMC(fk=fk_PF, N=10000, ESSrmin=1, resampling='multinomial', store_history=True, compute_moments=False, online_smoothing=None, verbose=False)

filtering.state_est(alg_PF, real_x, name="(CIR, SMC_10K)", xmin=-2, xmax=10)
plt.savefig(f"./Records/CIR{name}/SMC_10K_filtering_once.png")
plt.show()

evaluate.resi_plot(alg_PF, real_x, "SMC_10K")
plt.savefig(f"./Records/CIR{name}/SMC_10K_residuals_once.png")

#%% SMC_10K repeated 10
# average_state_est, average_running_time = repeated_simulation(fk_PF, R=10, alg_args=dict(default_args, N=10000))

#%% SMC_10K 50 repeats
# Ran in a single python file to avoid the inconsissence of Jupyter and multiprocessing
evaluate.running_result_evaluate(f"./Records/CIR{name}/running_result_10K_R50.npz", real_x, "SMC_10K")


#%% Particle filter SMC_100 (Truncated Normal proposal)
## %%timeit
# SMC_100, 100 particles each step
# 13.2 s ± 1.2 s per loop (mean ± std. dev. of 7 runs, 1 loop each) for N = 1000
alg_PF = particles.SMC(fk=fk_PF, N=N, ESSrmin=1, resampling='multinomial', store_history=True, compute_moments=False, online_smoothing=None, verbose=False)
filtering.state_est(alg_PF, real_x, name="(CIR, SMC_100)", xmin=-2, xmax=10)
plt.savefig(f"./Records/CIR{name}/SMC_100_filtering_once.png")

evaluate.resi_plot(alg_PF, real_x, "SMC_100")
plt.savefig(f"./Records/CIR{name}/SMC_100_residuals_once.png")

#%% SMC_100 repeated
# average_state_est, average_running_time = repeated_simulation(fk_PF, R=50)
evaluate.running_result_evaluate(f"./Records/CIR{name}/running_result_SMC100_R100.npz", real_x, "SMC_100")
plt.savefig(f"./Records/CIR{name}/SMC_100_average_est_{50}.png")

# %% Tail probability
all_particles = np.asarray(alg_PF.hist.X).reshape((100*100,1))
sns.distplot(all_particles)


#%% Modified SMC_100 R=100
evaluate.running_result_evaluate(f"./Records/CIR{name}/running_result_ModifiedSMC100_R100.npz", real_x, "ModifiedSMC_100")