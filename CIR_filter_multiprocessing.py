#%%
import datetime
import itertools
import multiprocessing
import os
import time
from functools import partial
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import particles
import seaborn as sns
from joblib import Parallel, delayed
from particles import state_space_models as ssm
from sklearn.metrics import mean_squared_error

import evaluate
from model.CIR import CIR, CIR_mod, CIR_plot

# %% Settings parameters
N = 100  # 5000
T = 1 * 10 ** 2  # 5 * 10 ** 4
R = 100  # Number of independent replications
my_seed = 3035802483
# Use same random seeds in R independent replications
MAX_INT_32 = np.iinfo(np.int32).max
seeds = [np.random.randint(MAX_INT_32) for i in range(R)]

#%% Load data from saved npz file
name = "20210204-225727"
temp = np.load(file=f"./Records/CIR{name}/data.npz")
real_x, real_y = temp.get("real_x"), temp.get("real_y")
fk_PF = ssm.GuidedPF(ssm=CIR(), data=real_y)
fk_MPF = ssm.GuidedPF(ssm=CIR_mod(s=50), data=real_y)
CIR_plot(real_x)


#%% Function for repeated filtering simulation
default_args = {
    "N": 100,
    "ESSrmin": 1,
    "resampling": "multinomial",
    "store_history": True,
    "compute_moments": False,
    "online_smoothing": None,
    "verbose": False,
}


def simulate_single(fk, alg_args, n=0):
    # print(f"#{os.getpid()} process start running.")
    t0 = time.time()
    alg = particles.SMC(fk=fk, **alg_args)
    alg.run()
    t1 = time.time()
    state_est = list(
        map(lambda i: np.average(alg.hist.X[i], weights=alg.hist.wgts[i].W), range(T))
    )
    print(f"#{os.getpid()} process finished, time costed: {t1-t0}")
    return state_est, t1 - t0


def repeated_simulation(fk, R=R, alg_args=default_args):
    # faster!
    num_cores = multiprocessing.cpu_count()
    f = partial(simulate_single, fk, alg_args)
    res = Parallel(n_jobs=num_cores)(delayed(f)(i) for i in range(R))
    return res


# An early version for multiprocessing simulation, using Pool.map() function in multiprocessing class
#
# def repeated_simulation_v2(fk, R=R, alg_args=default_args):
#     num_cores = multiprocessing.cpu_count()
#     p = Pool(processes=num_cores)
#     f = partial(simulate_single, fk, alg_args)
#     res = p.map(f, range(R))

#     return res


# %%
# The joblib version is faster for large number of repeats
if __name__ == "__main__":
    t0 = time.time()
    # SMC_10K
    # res = repeated_simulation(fk_PF, R=50, alg_args=dict(default_args, N=10000))
    res = res = repeated_simulation(fk_MPF, R=100)
    t1 = time.time()
    print(f"Joblib version time costed: {t1-t0}")
    np.savez(
        file=f"./Records/CIR{name}/running_result_ModifiedSMC100_R100",
        res=np.array(res),
    )

    evaluate.running_result_evaluate(
        f"./Records/CIR{name}/running_result_ModifiedSMC100_R100.npz",
        real_x,
        "ModifiedSMC_100",
    )

    # 20runs, 69.49423217773438 seconds
    # 50runs, 65.9800705909729 seconds
    # 50runs, SMC_10K, 2799.397787332535

    # For loop version
    # t0 = time.time()
    # for i in range(50):
    #     _, _ = simulate_single(fk_PF, default_args)
    # t1 = time.time()
    # print(f"For loop time costed: {t1-t0}")
    # 5 repeated run, 13.2 seconds
    # 20runs, 55.76940393447876 seconds
    # 50runs, 82.20281267166138 seconds

    # t0 = time.time()
    # res = repeated_simulation_v2(fk_PF, R=20)
    # t1 = time.time()
    # print(f"Multiprocessing time costed: {t1-t0}")
    # 10 runs, 41.604153871536255 seconds
    # 20 runs, 72.22063732147217
