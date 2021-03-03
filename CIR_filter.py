# -*- coding: utf-8 -*-
#%% Load packages
import datetime
import itertools
import os
import time
from functools import partial
from multiprocessing import Process, Queue
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import particles
import seaborn as sns
import statsmodels.api as sm
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from particles import state_space_models as ssm
from sklearn.metrics import mean_squared_error

import configparser
import evaluate
from filtering import state_est
from model.CIR import CIR, CIR_mod, CIR_plot

# %% Settings parameters
N = 100  # 5000
T = 1 * 10 ** 2  # 5 * 10 ** 4
R = 100  # Number of independent replications
my_seed = 3035802483
# Use same random seeds in R independent replications
MAX_INT_32 = np.iinfo(np.int32).max
seeds = [np.random.randint(MAX_INT_32) for i in range(R)]

tau = [0.25, 0.5, 1, 3, 5, 10]
#%% Define model and generate real data
# Run this cell to generate data or run the second cell below to load data generated and saved before.
cir = CIR()
real_x, real_y = cir.simulate(T)
fk_PF = ssm.GuidedPF(ssm=CIR(), data=real_y)
CIR_plot(real_x)

#%% Save the generated data
# Run this cell to save the newly generated data into the Records/CIRYYYYmmdd-HHMMSS folder.

name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(f"./Records/CIR{name}")  # Create folder

# Save the real_states plot
plt.savefig(f"./Records/CIR{name}/real_states.png")
# Save the generated data
np.savez(file=f"./Records/CIR{name}/data", real_x=real_x, real_y=real_y)  
# Save parameters of the model where data from.
config = configparser.ConfigParser()
config["PARAMETERS"] = {
    **cir.default_params,
    "Delta" : cir.Delta,
    "H"     : cir.H
}  
with open(f"./Records/CIR{name}/config.ini", "w") as f:
    config.write(f)

print(f"Data saved in folder CIR{name}.")

#%% Load data from saved npz file
name = "20210204-225727"  # Modify the name manually
loader = np.load(file=f"./Records/CIR{name}/data.npz")
real_x, real_y = loader.get("real_x"), loader.get("real_y")
fk_PF = ssm.GuidedPF(ssm=CIR(), data=real_y)

#%% Particle filter SMC_10K
alg_PF = particles.SMC(
    fk=fk_PF,
    N=10000,
    ESSrmin=1,
    resampling="multinomial",
    store_history=True,
    compute_moments=False,
    online_smoothing=None,
    verbose=False,
)
# Run the SMC_10K for one time.
state_est(alg_PF, real_x, name="(CIR, SMC_10K)", xmin=-2, xmax=10)
plt.savefig(f"./Records/CIR{name}/SMC_10K_filtering_once.png")
plt.show()
# Display the "residual" plot of the filtering result.
evaluate.resi_plot(alg_PF, real_x, "SMC_10K")
plt.savefig(f"./Records/CIR{name}/SMC_10K_residuals_once.png")

#%% SMC_10K 50 repeats
"""
Repeated simulation is conducted in `CIR_filter_multiprocessing.py` in a parallelling way, and the corresponding results are saved in a npz file.
"""
# Load and analyze the simulation result.
name = "20210204-225727"
evaluate.result_evaluate(
    f"./Records/CIR{name}/result_SMC_10K_R50.npz", real_x, "SMC_10K"
)
evaluate.tail_prob(f"./Records/CIR{name}/result_SMC_10K_R50.npz")

#%% Particle filter SMC_100
## %%timeit
alg_PF = particles.SMC(
    fk=fk_PF,
    N=N,  # 100
    ESSrmin=1,
    resampling="multinomial",
    store_history=True,
    compute_moments=False,
    online_smoothing=None,
    verbose=False,
)

# Run the SMC_100 for one time.
state_est(alg_PF, real_x, name="(CIR, SMC_100)", xmin=-2, xmax=10)
plt.savefig(f"./Records/CIR{name}/SMC_100_filtering_once.png")
# Display the "residual" plot of the filtering result.
evaluate.resi_plot(alg_PF, real_x, "SMC_100")
plt.savefig(f"./Records/CIR{name}/SMC_100_residuals_once.png")

#%% SMC_100 repeated
evaluate.result_evaluate(
    f"./Records/CIR{name}/result_SMC100_R500.npz", real_x, "SMC_100"
)
plt.savefig(f"./Records/CIR{name}/SMC_100_average_est_{500}.png")

#%% Modified SMC_100 R=100
evaluate.result_evaluate(
    f"./Records/CIR{name}/result_ModifiedSMC100_R500.npz",
    real_x,
    "ModifiedSMC_100",
)

#%% SMC_100 with t proposal density R=10
evaluate.result_evaluate(
    f"./Records/CIR{name}/SMCt100_R500/history.npz",
    real_x,
    "SMCt100_R500",
)
# evaluate.tail_prob(f"./Records/CIR{name}/result_SMCt100_R500.npz")

#%% Bootstrap filter
evaluate.result_evaluate(
    f"./Records/CIR{name}/Bootstrap100_R500/history.npz",
    real_x,
    "Bootstrap100_R500"
)

# %% tail probability 
file_list = ["Bootstrap100_R500",
             "SMC100_R500", 
            #  "ModifiedSMC100_R500", 
             "SMCt100_R500", 
             "SMC10K_R50"]
# file_list = [f"./Records/CIR{name}/"+f+"/history.npz" for f in file_list]
# evaluate.tail_prob_multi(file_list, real_x)

for name in file_list:
    evaluate.result_evaluate(f"./Records/CIR20210204-225727/{name}/history.npz", real_x, name)
    plt.savefig(f"./Records/CIR20210204-225727/{name}/state_est.png")

# %%
import importlib
importlib.reload(evaluate)
