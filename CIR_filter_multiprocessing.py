#%% Load packages
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
from model.CIR import CIR, CIR_mod, CIR_t, CIR_plot

# %% Settings parameters
N = 100  # 5000
T = 1 * 10 ** 2  # 5 * 10 ** 4
R = 100  # Number of independent replications
my_seed = 3035802483
# Use same random seeds in R independent replications
MAX_INT_32 = np.iinfo(np.int32).max
seeds = [np.random.randint(MAX_INT_32) for i in range(R)]

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


def repeated_simulation(fk, R=R, nprocs=0, alg_args=default_args):
    # faster!
    if nprocs == 0:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = min(multiprocessing.cpu_count(), abs(nprocs))
    f = partial(simulate_single, fk, alg_args)
    res = Parallel(n_jobs=num_cores)(delayed(f)(i) for i in range(R))
    return res

def multiSMC(fk_list, N=100, nruns=10, nprocs=0):
    for fk in fk_list.keys():
        t0 = time.time()
        res = repeated_simulation(fk_list.get(fk), nruns, nprocs)
        t1 = time.time()
        os.mkdir(f"./Records/CIR{name}/{fk}")
        file = f"./Records/CIR{name}/{fk}/history.npz"
        metadata = {
            "alg_name"      : fk,
            "num_rep"       : nruns,
            "timestamp"     : str(datetime.datetime.now()),
            "file_path"     : file
        }
        # Save the result
        np.savez(
            file = file,
            res = np.array(res),
            meta = metadata
        )
        print(f"[Finish] Total time costed: {t1-t0}\nHistory saved at {file}")

# An early version for multiprocessing simulation, using Pool.map() function in multiprocessing class
#
# def repeated_simulation_v2(fk, R=R, alg_args=default_args):
#     num_cores = multiprocessing.cpu_count()
#     p = Pool(processes=num_cores)
#     f = partial(simulate_single, fk, alg_args)
#     res = p.map(f, range(R))

#     return res


# %% Run the multiprocessing simulation task
# The joblib version is faster for large number of repeats
if __name__ == "__main__":
    # Load data from saved npz file and initilize fk models
    name = "20210225-111533"

    loader = np.load(file=f"./Records/CIR{name}/data.npz")
    real_x, real_y, real_y2 = loader.get("real_x"), loader.get("real_y"), loader.get("real_y2")
    CIR_plot(real_x)
    
    # For the univariate case simulation
    # fk_list = {
    #     "fk_boot" : ssm.Bootstrap(ssm=CIR(), data=real_y),
    #     "fk_PF" : ssm.GuidedPF(ssm=CIR(), data=real_y),
    #     "fk_MPF" : ssm.GuidedPF(ssm=CIR_mod(s=50), data=real_y),
    #     "fk_PF_t" : ssm.GuidedPF(ssm=CIR_t(), data=real_y)
    # }

    # For the multivariate case simulation
    tau = np.array([0.25, 0.5, 1, 3, 5, 10])
    fk_list = {
        "bootstrap_H10": ssm.Bootstrap(ssm=CIR(tau=tau, H=10), data=real_y),
        "bootstrap_H1" : ssm.Bootstrap(ssm=CIR(tau=tau, H=1), data=real_y2),
        "SMC_H10"      : ssm.GuidedPF(ssm=CIR(tau=tau, H=10), data=real_y),
        "SMC_H1"       : ssm.GuidedPF(ssm=CIR(tau=tau, H=1), data=real_y2),
        "SMCt_H10"     : ssm.GuidedPF(ssm=CIR_t(tau=tau, H=10), data=real_y),
        "SMCt_H1"      : ssm.GuidedPF(ssm=CIR_t(tau=tau, H=1), data=real_y2),
}

    multiSMC(fk_list, nruns=500)

    """
    fk_boot = ssm.Bootstrap(ssm=CIR(), data=real_y)
    fk_PF = ssm.GuidedPF(ssm=CIR(), data=real_y)
    fk_MPF = ssm.GuidedPF(ssm=CIR_mod(s=50), data=real_y)
    fk_PF_t = ssm.GuidedPF(ssm=CIR_t(), data=real_y)

    # Start expirment ===================================================
    alg_name = "Bootstrap100_R500"
    R = 500

    t0 = time.time()
    # res = repeated_simulation(fk_PF, R=50, alg_args=dict(default_args, N=10000))  # SMC_10K
    res = repeated_simulation(fk_boot, R=R)
    t1 = time.time()

    file = f"./Records/CIR{name}/{alg_name}/history.npz"
    metadata = {
        "alg_name"      : alg_name,
        "num_rep"       : R,
        "timestamp"     : str(datetime.datetime.now()),
        "file_path"     : file
    }
    # Save the result
    np.savez(
        file = file,
        res = np.array(res),
        meta = metadata
    )
    print(f"[Finish] Total time costed: {t1-t0}\nHistory saved at {file}")

    # Evaluate the estimating result
    evaluate.result_evaluate(
        file,
        real_x,
        alg_name,
    )
    evaluate.tail_prob(file)
    plt.show()
    """
