#%% Load packages
import numpy as np
from model import CIR
from model.CIR import post_mean
from model.CIR import CIR_config
import MultiPropPF
import matplotlib.pyplot as plt
import particles
from particles import state_space_models as ssm
import time
import importlib
import datetime
import configparser
import  pandas as pd
import evaluate
from evaluate import observation_plot

#%% Load data from saved npz file
# record_id = "20210330-181131"
record_id = "20210331-183312"
record_id = "20210401-170824"  # This dataset contains no outliers in observations
loader = np.load(file=f"./Records/CIR{record_id}/data.npz")
real_x, real_y = loader.get("real_x"), loader.get("real_y")

# Generating New Dataset =============================================================================================
#%% Parameters
cir_params = {"kappa":0.169, "theta":6.56, "sigma":0.321, 'lam':-0.0201,
              "Delta":1/12, 'tau':np.array([0.25, 1, 3, 5, 10])}  # (Neslihanoglu and Date 2018)
cir_params = {"kappa":0.1862, "theta":0.0654, "sigma":0.0481, 'lam':-0.074105,
              "Delta":1/12/30, 'tau':np.array([0.25, 1, 3, 5, 10])}  # (Rossi 2010)

T = 100
tau = np.array([0.25, 0.5, 1, 3, 5, 10])
H = 1

# Def 1 of SNR
# SNR = cir_params['sigma']**2 * cir_params['theta'] / H
H = cir_params['sigma']**2 * cir_params['theta'] / 0.5; SNR = 0.5  # SNR = 0.5
H = cir_params['sigma']**2 * cir_params['theta'] / 1; SNR = 1   # SNR = 1
H = cir_params['sigma']**2 * cir_params['theta'] / 5; SNR = 5   # SNR = 5
H = cir_params['sigma']**2 * cir_params['theta'] / 10; SNR = 10  # SNR = 10

# Def 2 of SNR (in doubt)  var(noncentra chi^2) / H
def cir_snr(parms, h):
    '''
    "average" variance of the ncx2 distributed transition density / variance of normal distributed observation density
    '''
    k = 4*parms['kappa']*parms['theta']/parms['sigma']**2
    c = 2 * parms['kappa'] / (parms['sigma']**2 * (1-np.exp(-parms['kappa']*parms['Delta'])))
    l = 2*c*np.exp(-parms['kappa']*parms['Delta'])*parms['Delta']  # Use Delta to replace x_{t-1} in l, in order to repersent an average level
    var_ncx2 = 2*(k+2*l) / (2*c)**2
    return var_ncx2/h
cir_snr(cir_params, H)
# (k+2l)/2c^2 = k/2c^2 + l/c^2

#%% Define CIR model and generate data
# cir = CIR.CIR(tau=tau, H=H, kappa=kappa, theta=theta, sigma=sigma)
cir = CIR.CIR(**cir_params, H=H)
real_x, real_y = cir.simulate(T=T)
CIR.CIR_plot(real_x)
observation_plot(real_y, cir_params['tau'])

# Generate observations with outliers (from t distribution)
# real_y = CIR.CIR_t(tau=tau, H=H).simulate_given_x(real_x)
real_y = CIR.CIR_t(tau=tau, H=H, emotion=True, nc=1.3).simulate_given_x(real_x)
observation_plot(real_y, tau)
pd.DataFrame(np.vstack(real_y)).describe()

#%% Save data into npz file
is_new_data = False
if is_new_data:
    import os
    # record_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    record_id = f"_{SNR}"
    os.mkdir(f"./Records/CIR{record_id}")  # Create folder
    # Save the real_states plot
    CIR.CIR_plot(real_x)
    plt.savefig(f"./Records/CIR{record_id}/real_states.png")
    # Save the real_observation plot
    observation_plot(real_y, cir_params['tau'])
    plt.savefig(f"./Records/CIR{record_id}/real_y.png")
    # Save the generated data
    np.savez(file=f"./Records/CIR{record_id}/data", real_x=real_x, real_y=real_y)

    CIR_config(record_id).generate(cir)

    ''' Old codes using configparser
    config = configparser.ConfigParser()
    config["PARAMETERS"] = {
        # **cir.default_params,
        'kappa': kappa,
        'theta': theta ,
        'sigma': sigma,
        'lam': lam
    }
    config["OTHERS"] = {
        "Delta": cir.Delta,
        "tau": cir.tau,
        "H": cir.H
    }
    with open(f"./Records/CIR{record_id}/config.ini", "w") as f:
        config.write(f)
    '''
    print(f"Data saved in folder CIR{record_id}.")

# End Generate New Dataset ===========================================================================================
# observation_plot(real_y, tau)

#%% Default parameters for the SMC object
default_args = {
    'N': 100,
    'ESSrmin': 1,
    'resampling': 'multinomial',
    'store_history': True,
    'moments' : CIR.post_mean,
    # 'compute_moments': False,
    'online_smoothing': None,
    'verbose': False
}

#%% Generate 'real states` (population states) from a particle filter with huge amoung of particles.
# Update @20210324: code for generating posterior mean and aid_x are moved to CIR_filter_multiprocessing.py.

# RESULT ANALYSIS ====================================================================================================
#%% Analuze saved result of CIR_filter_multiprocessing.multiSMC
record_id = "20210330-181131"
record_id = "20210331-183312"  # New version outlier
record_id = "20210401-170824"  # This dataset contains no outliers in observations
record_id = "_10"

loader = np.load(file=f"./Records/CIR{record_id}/data.npz")
real_x, real_y = loader.get("real_x"), loader.get("real_y")

loader = np.load(file=f"./Records/CIR{record_id}/aid_x_boot.npz")
aid_x = loader.get('aid_x')
aid_post_mean = loader.get('aid_post_mean')
PX_quantiles = loader.get('PX_quantiles')
config = CIR_config(record_id).load()
cir_args = config['MODEL']['PARAMS']
tau, H = config['MODEL']['OBS']['tau'], config['MODEL']['OBS']['H']


# alg_list = ['bootstrap', 'SMC', 'SMCt', 'SMC_mod', 'MultiPropPF', 'MultiPropPF2', 'MultiPropPF3', 'MultiPropPF4', 'MultiPropPF5']
alg_list = ['bootstrap', 'SMC', 'SMCt', 'SMC_mod']#, 'MultiPropPF4', 'MultiPropPF5','MultiPropPF6', 'MultiPropPF7']
alg_list = ['bootstrap', 'SMC', 'SMCt', 'normal+q', 't+q', 'normal+q(l)', 'normal+q(r)']

alg_list = ['bootstrap', 'SMC_mod', 'SMC', 'SMCt', 'normal+q811.99', 't5+q811.95']#, 't5+q811.9', 't3+q811.9']
file_list = [f"./Records/CIR{record_id}/" + f + "/history.npz" for f in alg_list]
average_mse = pd.DataFrame(np.zeros(shape=(len(alg_list),2)), index=alg_list, columns=["average_mse", "running_time"])

for name in alg_list:
    # average_mse.loc[name] = evaluate.result_evaluate(f"./Records/CIR{record_id}/{name}/history.npz", real_x, name=name)
    average_mse.loc[name] = evaluate.result_evaluate(f"./Records/CIR{record_id}/{name}/history.npz", aid_post_mean, name=name)
    plt.savefig(f"./Records/CIR{record_id}/{name}/state_est.png")

print(average_mse.iloc[:,0])

evaluate.log_mset_plot(file_list, real_x)
plt.savefig(f"./Records/CIR{record_id}/log_mse.png")


#%% Tail probability analysis
alpha = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75,  0.95, 0.99, 0.999]
alpha = [0.0001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
alpha = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
alpha = reduce(np.append, [np.power(10., [-8, -5, -3, -2]),
                           np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
                           1-np.flip(np.power(10., [-8, -5, -3, -2]))])

# 1. Display the approximated theoretical quantiles
plt.subplots()
lines = plt.plot(np.quantile(aid_x, alpha, axis=1).T)
plt.legend(lines, alpha)
plt.show()

# Add shadows arond the lines, to illustrate the distribution of quantiles estimation estimated by our algorithms, use
# different alpha to represent densities. (kernel density plot around lines)

# 2. Compute MSEs between quantiles estimated by algorithms and the standard.
# importlib.reload(evaluate)
alpha = [alpha[0]]
t0 = time.time()
tail_analysis = evaluate.tail_prob_multi(file_list, aid_x, alpha=alpha)
t1 = time.time()
print(t1 - t0)
print(tail_analysis.to_string())

# tail_analysis = evaluate.tail_prob_multi(file_list, aid_x, alpha=alpha, real_quantiles=PX_quantiles.T)

# Assisting analysis ================================================================================================
# ===================================================================================================================
# Compare the distribution of particles generated by particle filter algorithms and aid_x used as standard ===========
# See the extreme tail part, whether the kde density of xs could cover density of aid_x
state_estimates, xs, _ = evaluate.load_hist(f"./Records/CIR{record_id}/MultiPropPF5/history.npz")
importlib.reload(evaluate)
evaluate.PXCheck(xs, aid_x, 15)

# Distribution of estimated quantiles vs real quantile from aid_x =================================================
# Cross-sectional analysis
importlib.reload(evaluate)
alpha = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
evaluate.q_dists(file_list, aid_x, alpha, t=30)

# Trajectories of average estimated quantiles and "real" quantiles ====================================================
importlib.reload(evaluate)
evaluate.q_trajectories(file_list, aid_x, alpha)


# Use real quantiles of the non-central chi-square distribution (PX) =================================================
# WRONG! Filtering mean estimated by particle filter is the posteriror mean of states given observation.
# Cannot compare the posterior distribution with transistion density.

state_estimates, xs, _ = evaluate.load_hist(f"./Records/CIR{record_id}/bootstrap/history.npz")
state_estimates, xs, _ = evaluate.load_hist(f"./Records/CIR{record_id}/boot+q/history.npz")
# PX_quantiles = np.array([CIR.CIR(tau=tau, H=H, **cir_args).PX(t, x).ppf(alpha) for t, x in enumerate(real_x[:-1], start=1)])
quantiles = np.quantile(xs, alpha, axis=2).reshape((100, len(alpha), T))
mset = np.mean(np.square((quantiles[:,:,1:] - PX_quantiles.T)), axis=0)
np.mean(mset, axis=1)

# Compare the PX and proposals in CIR class ==========================================================================
# The two proposal densities are correct in the aspect of relative position. Their means are set same as expect.
import model.distributions
from matplotlib.animation import FuncAnimation
importlib.reload(evaluate)
cir = CIR.CIR(tau=tau, H=H, **cir_args)
# ts = np.arrange()
t = 20
dist_args = {'t': t, 'xp': aid_x[t-1,:].mean(), 'data': real_y}
fig, ax = plt.subplots()
evaluate.dist_cmp({
    'px': cir.PX(dist_args['t'], dist_args['xp']),
    'norm': cir.proposal_normal(**dist_args),
    't': cir.proposal_t(**dist_args) # model.distributions.t_nn(df=30, loc=dist_norm.mu, scale=dist_norm.sigma)
    # 'tq': cir.proposal_t_quantile(**dist_args)
}, title=f"t={t}", ax=ax)

# Make animation
def update(t):
    dist_args = {'t': t, 'xp': real_x[t-1], 'data': real_y}
    # dist_args = {'t': t, 'xp': aid_x[t - 1, :].mean(), 'data': real_y}
    ax.clear()
    evaluate.dist_cmp({
        'px': cir.PX(dist_args['t'], dist_args['xp']),
        'norm': cir.proposal_normal(**dist_args),
        't': cir.proposal_t(**dist_args)  # model.distributions.t_nn(df=30, loc=dist_norm.mu, scale=dist_norm.sigma)
        # 'tq': cir.proposal_t_quantile(**dist_args)
    }, title=f"Transition density and proposals at t={t}", ax=ax)

fig, ax = plt.subplots()
with np.errstate(divide='ignore', invalid='ignore'):
    anim = FuncAnimation(fig, update, frames=np.arange(1, 100), interval=300)
    # anim.save('compare_densities2.gif', dpi=80, writer='imagemagick')
    plt.show()

# Trajectories of means of PX and proposals ===========================================================================
cir = CIR.CIR(tau=tau, H=H, **cir_args)  # CIR with normal proposal
alg = particles.SMC(fk=ssm.GuidedPF(ssm=cir, data=real_y), **default_args)
alg.run()
xh = alg.summaries.moments  # Estimated posterior mean
dists = pd.DataFrame({'px': np.array([cir.PX(t, xp).stats[0] for t, xp in enumerate(real_x[:99], start=1)]).reshape((99,)),
        'norm': np.array([cir.proposal_normal(t, xp, real_y).mu for t, xp in enumerate(real_x[:99], start=1)]).reshape((99,)),
        't': np.array([cir.proposal_t(t, xp, real_y).stats[0] for t, xp in enumerate(real_x[:99], start=1)]).reshape((99,))
         })
dists.plot()


# Proposals and their tails ===========================================================================================
# Cross-sectional analysis
importlib.reload(CIR)
cir = CIR.CIR(tau=tau, H=H, **cir_args)  # CIR with normal proposal
alg = particles.SMC(fk=ssm.GuidedPF(ssm=cir, data=real_y), **default_args)
alg.run()
xh = alg.summaries.moments  # Estimated posterior mean

t = 30
xs = np.arange(0, 5, 0.01)
# fig, ax = plt.subplots(nrows=2, figsize=(8, 12))
# ax = ax.flatten()
# ax[0].plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='normal', **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)), '-b')
# ax[0].plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='normalql', **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)), '--r')
# ax[0].plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='normalq', **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)), '--g')
#
#
# ax[1].plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='t', **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)), '-b')
# ax[1].plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='tql', **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)), '--r')
# ax[1].plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='tq', **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)), '--g')

plt.subplots()
prop = ('normal', 't')[1]
tdf = 2; ptq = 0.95  # alg: t3+q811.9
# tdf = 5; ptq = 0.95  # alg: t5+q811.95
# ptq = 0.99  # alg: normal+q811.99
plt.plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='t', **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)), '-', label='optimal proposal')
plt.plot(xs, 0.8 * np.exp(CIR.CIR(tau=tau, H=H, proposal=prop, tdf=tdf, ptq=ptq,  **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)) +
             0.1 * np.exp(CIR.CIR(tau=tau, H=H, proposal=prop+'ql' ,tdf=tdf, ptq=ptq,  **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)) +
             0.1 * np.exp(CIR.CIR(tau=tau, H=H, proposal=prop+'q', tdf=tdf, ptq=ptq,  **cir_args).proposal(t, xh[t-1], real_y).logpdf(xs)),
         '-', label='mixture proposal')
# plt.plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='t', **cir_args).proposal_boot(t, xh[t-1], real_y).logpdf(xs)), '-', label='transition density (BF proposal)')
sns.kdeplot(aid_x[t,:], label='posterior density')
plt.title(f"Compare proposals and posterior density")
plt.legend()

plt.subplots()
plt.plot(xs, np.exp(CIR.CIR(tau=tau, H=H, proposal='t', **cir_args).proposal_boot(t, xh[t-1], real_y).logpdf(xs)), '-', label='transition density')
sns.kdeplot(aid_x[t,:], label='posterior density')
plt.title(f"Compare proposals and posterior density at t={t} (bootstrap)")
plt.legend()

# Algorithm illustrate step by step ===================================================================================
# KDE plot of approximate posterior distribution, particles and their weights
fk = MultiPropPF.MultiPropFK(ssm=CIR.CIR(tau=tau, H=H, **cir_args), data=real_y,
                                             proposals={'normal': 0.9, 'normalq': 0.05, 'normalql': 0.05})
alg = MultiPropPF.MultiPropPF(fk=fk, **default_args)
evaluate.alg_illustrate_next(alg, aid_x)

def update(t):
    dist_args = {'t': t, 'xp': real_x[t-1], 'data': real_y}
    # dist_args = {'t': t, 'xp': aid_x[t - 1, :].mean(), 'data': real_y}
    ax.clear()
    evaluate.alg_illustrate_next(alg, aid_x, ax=ax)

fig, ax = plt.subplots()
with np.errstate(divide='ignore', invalid='ignore'):
    anim = FuncAnimation(fig, update, frames=np.arange(1, 100), interval=300)
    # anim.save('compare_densities2.gif', dpi=80, writer='imagemagick')
    plt.show()


# Compare exact transition density with the approximate normal in (Neslihanoglu 2018) ================================
from scipy.stats import truncnorm
cir = CIR.CIR(tau=tau, H=H, **cir_args)
alg = particles.SMC(fk=ssm.GuidedPF(ssm=cir, data=real_y), **default_args)
alg.run()
xh = alg.summaries.moments  # Estimated posterior mean

xs = np.arange(0, 5, 0.01)

def update(t):
    mu_x = cir.k / 2 / cir.c + np.exp(-cir.kappa*cir.Delta)*xs[t-1]
    var_x = cir.k / 2 / cir.c**2 + 2 * np.exp(-cir.kappa*cir.Delta) * xs[t-1] / cir.c
    ax.clear()
    plt.plot(xs, np.exp(cir.PX(t, xs[t-1]).logpdf(xs)), '-r', label='exact transition density')
    plt.plot(xs, truncnorm(loc=mu_x, scale=np.sqrt(var_x), a=-mu_x / np.sqrt(var_x), b=np.inf).pdf(xs),
             label='normal approximation')
    ax.set(title=f't={t}', xlabel='x', ylabel='density')
    plt.legend()

fig, ax = plt.subplots()
with np.errstate(divide='ignore', invalid='ignore'):
    anim = FuncAnimation(fig, update, frames=np.arange(1, 100), interval=300)
    # anim.save('compare_densities2.gif', dpi=80, writer='imagemagick')
    plt.show()

update(30)

fig, ax = plt.subplots(ncols=2)
ax[0].plot(stats.skew(aid_x, axis=1))
ax[0].set(title='skewness', xlabel='T', ylabel='skewness')
ax[1].plot(stats.kurtosis(aid_x, axis=1))
ax[1].set(title='kurtosis', xlabel='T', ylabel='kurtosis')
# ================================ Unused historic code ===================================
'''
Long log ago, I use the particles.multiSMC method to run several particle filters. The corresponding results are stored
in one file, results_xxx.npz, due to the limitation of this method, which is also very slow (can only use one core). 
Finally I gave up this method, and following codes for analyzing the results_xxx.npz results were swept into the dustbin
of history.
'''

'''
#%% Anayze saved result of particles.multiSMC
results = np.load(f"./Records/CIR{record_id}/results_011216.npz", allow_pickle=True).get("results")

state_est = {results[i].get("fk"): 
    np.array([list(map(lambda i: np.average(r.get("particles")[i], weights=r.get("weights")[i]), range(T)))
    for r in results[i::4]]) 
    for i in range(4)
}

res = {
    fk: {
        "average_state_est": np.mean(state_est.get(fk), axis=0),
        "state_std": np.std(state_est.get(fk), axis=0)
    }
    for fk in state_est.keys()
}

for fk in res.keys():
    evaluate.average_state_est_plot(res[fk]["average_state_est"], res[fk]["state_std"], real_x, fk)

'''
#%%
# gamma = np.sqrt((kappa+lam)**2 + 2*sigma**2)

# Btau = 1/tau * (2*(np.exp(gamma*tau)-1)) / (2*gamma+(kappa+lam+gamma)*(np.exp(gamma*tau)-1))

# Atau = (2*kappa*theta/sigma**2 + 1) / tau * np.log((2*gamma*np.exp(tau*(kappa+lam+gamma)/2)) / (2*gamma+(kappa+lam+gamma)*(np.exp(gamma*tau)-1)))

# x = 3.21

# mu = -Atau + Btau*x
# sigma = np.diag(H)
