#%% Load packages
import datetime
import multiprocessing
import os
import time
import configparser
from loguru import logger
from functools import partial

import numpy as np
import particles
import MultiPropPF
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed

from particles import state_space_models as ssm

from model.CIR import CIR, CIR_plot, CIR_mod, CIR_config, post_mean


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
    'N': 100,
    'ESSrmin': 1,
    'resampling': 'multinomial',
    'store_history': True,
    'moments' : post_mean,
    # 'compute_moments': False,
    'online_smoothing': None,
    'verbose': False
}

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def simulate_single(fk, alg_args, n=0):
    # print(f"#{os.getpid()} process start running.")
    t0 = time.time()
    if isinstance(fk, MultiPropPF.MultiPropFK):
        alg = MultiPropPF.MultiPropPF(fk=fk, **alg_args)
    else:
        alg = particles.SMC(fk=fk, **alg_args)
    alg.run()
    t1 = time.time()
    # state_est = list(
    #     map(lambda i: np.average(alg.hist.X[i], weights=alg.hist.wgts[i].W), range(T))
    # )
    state_est = alg.summaries.moments
    xs = alg.hist.X
    print(f"#{os.getpid()} done ({t1-t0}s)")
    return state_est, xs, t1 - t0


def repeated_simulation(fk, R=R, nprocs=0, alg_args=default_args):
    # faster!
    if nprocs == 0:
        num_cores = multiprocessing.cpu_count()  # 8
    else:
        num_cores = min(multiprocessing.cpu_count(), abs(nprocs))
    f = partial(simulate_single, fk, alg_args)
    with tqdm_joblib(tqdm(desc="Simulation", total=R)) as progress_bar:
        res = Parallel(n_jobs=num_cores)(delayed(f)(i) for i in range(R))
    return res

def multiSMC(fk_list, N=100, nruns=10, nprocs=0):
    logger.info("The multiSMC task starts.\nSummary: {} algorithms, with {} repeats and {} particles in each time step.",
                len(fk_list), nruns, N)
    for fk in fk_list.keys():
        logger.info("Task {} starts..", fk)
        os.mkdir(f"./Records/CIR{record_id}/{fk}")
        t0 = time.time()
        res = repeated_simulation(fk_list.get(fk), nruns, nprocs, alg_args={**default_args, 'N': N})
        t1 = time.time()

        file = f"./Records/CIR{record_id}/{fk}/history.npz"
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
        logger.success("Task {} ends, time costed is {}.\nHistory saved at {}.", fk, t1-t0, file)
        # print(f"[Finish] Total time costed: {t1-t0}\nHistory saved at {file}")
    logger.success("The multiSMC task ends.")

# An early version for multiprocessing simulation, using Pool.map() function in multiprocessing class
#
# def repeated_simulation_v2(fk, R=R, alg_args=default_args):
#     num_cores = multiprocessing.cpu_count()
#     p = Pool(processes=num_cores)
#     f = partial(simulate_single, fk, alg_args)
#     res = p.map(f, range(R))

#     return res

def generate_aid_x(record_id, data, cir_args, tau, H, N=100000):
    logger.info("Start to generate posterior mean used as benchmark.")
    aid_ssm = CIR(tau=tau, H=H, **cir_args)
    aid_alg = particles.SMC(ssm.Bootstrap(ssm=aid_ssm, data=data), **{**default_args, 'N': N})
    aid_alg.run()
    store = {
        'aid_x': aid_alg.hist.X,
        'aid_post_mean': aid_alg.summaries.moments
    }
    np.savez(file=f"./Records/CIR{record_id}/aid_x_boot", **store)
    logger.info("Posterior mean successfully generated.")



# %% Run the multiprocessing simulation task
# The joblib version is faster for large number of repeats
if __name__ == "__main__":
    # Load data and parameters =======================================================================================
    # record_id = "20210331-183312"
    # record_id = "20210401-170824"
    record_id = "_0.5"
    nruns = 100
    N = 100

    loader = np.load(file=f"./Records/CIR{record_id}/data.npz")
    real_x, real_y = loader.get("real_x"), loader.get("real_y")

    is_old_config = False
    if is_old_config:
        tau = np.array([0.25, 0.5, 1, 3, 5, 10])
        H = 0.5
        config = configparser.ConfigParser()
        config.read(f"./Records/CIR{record_id}/config.ini")
        cir_args = {key: config['PARAMETERS'].getfloat(key) for key in config['PARAMETERS'].keys()}
        logger.info("Data lodaded successfully. Shape of states x: {}; Shape of observations y: {}.", real_x.shape, real_y.shape)
    else:
        config = CIR_config(record_id).load()
        cir_args = config['MODEL']['PARAMS']
        tau, H = config['MODEL']['OBS']['tau'], config['MODEL']['OBS']['H']

    # Define Feynman-Kac models ======================================================================================
    fk_list = {
        "bootstrap_60k"     : ssm.Bootstrap(ssm=CIR(tau=tau, H=H, **cir_args), data=real_y),
        # "SMC"           : ssm.GuidedPF(ssm=CIR(tau=tau, H=H, **cir_args), data=real_y),
        # "SMCt"         : ssm.GuidedPF(ssm=CIR(tau=tau, H=H, **cir_args, proposal='t', tdf=5), data=real_y),
        # "SMC_mod"       : ssm.GuidedPF(ssm=CIR_mod(tau=tau, H=H, s=50, proposal='boot', **cir_args), data=real_y),
        # #
        # "normal+q811.99": MultiPropPF.MultiPropFK(ssm=CIR(tau=tau, H=H, ptp=0.99, **cir_args), data=real_y,
        #                                      proposals={'normal': 0.8, 'normalq': 0.1, 'normalql': 0.1}),
        # "normal+tq": MultiPropPF.MultiPropFK(ssm=CIR(tau=tau, H=H, tdf=3, **cir_args), data=real_y,
        #                                      proposals={'normal': 0.9, 'tq': 0.05, 'tql': 0.05}),
        # 为normal插上t的翅膀

        # "normal+q3"    : MultiPropPF.MultiPropFK(ssm=CIR(tau=tau, H=H), data=real_y,
        #                                             proposals={'normal':0.8, 'normalq':0.1, 'normalql':0.1}),
        # 2021-04-08 3:10 enlarge the proportion of right sided tail, expect better performance on extreme quantiles estimate
        #
        # "t5+q811.95": MultiPropPF.MultiPropFK(ssm=CIR(tau=tau, H=H, tdf=5, ptq=0.95, **cir_args), data=real_y,
        #                                       proposals={'t': 0.8, 'tq': 0.1, 'tql': 0.1}),

        # "t3+q811.9"         : MultiPropPF.MultiPropFK(ssm=CIR(tau=tau, H=H, tdf=3, ptq=0.9, **cir_args), data=real_y,
        #                                          proposals={'t': 0.8, 'tq': 0.1, 'tql': 0.1})
    }

    # 在不眠的夜里，呆坐在屏幕前，期待simulation出满意结果的阵子，是焦虑的时光里的短暂慰藉吗？
    # Run multiple particle filters ===============================================================================
    N = 60000
    multiSMC(fk_list, nruns=nruns, N=N, nprocs=6)


    # Generate aid quantities ======================================================================================
    # %% Generate 'real states` (population states) from a particle filter with huge amoung of particles
    new_aid_x = False  # whether compute new aid x
    if new_aid_x:
        # alpha = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
        logger.info("Start to generate posterior mean used as benchmark.")
        aid_ssm = CIR(tau=tau, H=H, proposal='t', tdf=10, **cir_args)
        # aid_alg = particles.SMC(fk=ssm.GuidedPF(ssm=aid_ssm, data=real_y),
        #                         **{**default_args, "N": 10000})
        aid_alg = particles.SMC(ssm.Bootstrap(ssm=aid_ssm, data=real_y), **{**default_args, 'N':100000})
        t0 = time.time()
        aid_alg.run()
        t1 = time.time()
        print(f"{t1-t0}s")
        # aid_x = np.concatenate(aid_alg.hist.X, axis=0)
        store = {
            'aid_x': aid_alg.hist.X,
            'aid_post_mean': aid_alg.summaries.moments
            # 'aid_quantiles': np.quantile(np.array(aid_alg.hist.X,), alpha, axis=1)
            # 'PX_quantiles': np.array([aid_ssm.PX(t, x).ppf(alpha) for t, x in enumerate(real_x[:-1], start=1)])
        }

        np.savez(file=f"./Records/CIR{record_id}/aid_x_boot", **store)
        logger.info("Posterior mean successfully generated.")
        # aid_x_boot: use bootstrap filter to approximate posterior density $p(x_{t}|y_{t})$

    logger.info("Program finished. Bye.")
