# Todo: Add logger for evaluating functions

import numba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import itertools
from loguru import logger
from sklearn.metrics import mean_squared_error


def load_hist(file):
    """
    Load the saved algorithm running history in npz format.

    Args:
        file (string): path of the history file.

    Returns:
        1. Estimated states
        2. Running time
    """
    loader = np.load(file=file, allow_pickle=True)
    if loader.get("meta"):
        meta = loader.get("meta").item()
        logger.info("History of {} loaded successfully. Created time: {}", meta['alg_name'], meta['timestamp'])
        # print(f"[INFO] History loaded\nAlgorithm: {meta['alg_name']}\nCreated time: {meta['timestamp']}\n")
    else:
        logger.info("History file {} loaded successfully", file)
    res = loader.get("res")
    state_estimates, xs, running_time = np.stack(res[:,0], axis=0), np.stack(np.array(res)[:,1], axis=0), res[:,2]
    return state_estimates, xs, running_time


def resi_plot(alg, real_x, name=""):
    """
    Calculate and plot residuals of the estimated states.
    
    Args:
        alg (particles.SMC)  : a already ran SMC object which stores the particles and weights.
        real_x (numpy.array) : real states
        name (string)        : name of this SMC algorithm
    """
    T = len(alg.fk.data)
    state_estimates = list(map(lambda i: np.average(alg.hist.X[i], weights=alg.hist.wgts[i].W), range(T)))
    fig = plt.figure(figsize = (8,4))
    real_x = np.array(real_x).reshape((len(real_x), ))
    plt.plot([state_estimates[i] - real_x[i] for i in range(T)], '.')
    plt.hlines(y=0, xmin=0, xmax=T, linestyles="dashed")
    plt.xlabel("time index")
    plt.ylabel(r"$\hat{x}_t - x_t$")
    mse = mean_squared_error(state_estimates, real_x) #list(itertools.chain.from_iterable(real_x)))
    if name:
        plt.title(f"{name}: Estimated state - Real state (MSE={mse:.3})")
    else:
        plt.title(f"Estimated state - Real state (MSE={mse:.3})")
    # plt.show()

def average_state_est_plot(average_state_est, state_std, real_x, T: int = None, name: str = ""):
    """
    Plot true states and average estimated states from a repeated running algorithm.

    Args:
        average_state_est ([type]): [description]
        real_x ([type]): real_states or estimated posterior mean of states
        R ([type]): number of repetations
        T (int, optional): first T time steps to plot, default all.
        name (str, optional): [description]. Defaults to "".
    
    Refer to Figure 2 of Neslihanoglu, S., & Date, P. (2019).
    Neslihanoglu, S., & Date, P. (2019). A modified sequential Monte Carlo procedure for the efficient recursive estimation of extreme quantiles. Journal of Forecasting, 38(5), 390-399.
    """
    _, ax = plt.subplots(figsize = (8,6))
    if not T:
        T = len(real_x)
    real_x, average_state_est = real_x[:T], average_state_est[:T]

    plt.plot(real_x, color="C0", label="True State")
    plt.plot(average_state_est, linestyle="dashed", color="C0", label=f"{name} State Estimate")

    lower = average_state_est - state_std * 1.96
    upper = average_state_est + state_std * 1.96
    plt.fill_between(np.linspace(0, T, T), lower, upper, alpha=0.2)
    
    ax.set(title = f"Average State Estimates from {name} Simulation",
        xlabel = r"time index ($t$)",
        ylabel = r"Average state estimates ($\bar{\hat{x}}_t$)")

    plt.legend()
    logger.info("average_state_est_plot for {} finished", name)

def result_evaluate(file, real_x, name=""):
    """
    Evaluate running result, calculate the average running time of R repeatitions, mean squared error of the average
    state estimates, and plot the average state estimates with real states.

    Args:
        file (string): path of the result file.
        real_x (numpy.array): real_states
        name (str, optional): description of the algorithm. Defaults to "".
    """
    state_estimates, _, running_time = load_hist(file)
    # state_estimates (N, T)
    # R = len(running_time)
    average_running_time = np.mean(running_time)
    average_state_est = np.mean(state_estimates, axis=0)
    state_std = np.std(state_estimates, axis=0)
    # mse = mean_squared_error(average_state_est, real_x) #list(itertools.chain.from_iterable(real_x)))

    mses = (np.square(state_estimates - real_x.reshape(1,len(real_x)))).mean(axis=0)
    mseb = mses.mean()
    # $\overline{\mathrm{MSE}}=\frac{1}{TR}\sum_{t=1}^{T}\sum_{r=1}^{R}\left(\hat{\mu}_{t}^{(r)}-\mu_{t}\right)^{2}$

    logger.debug(f"The average running time of {name} is {average_running_time}, average MSE is {mseb}.")
#     print(f"The average running time is {average_running_time}\n\
# MSE: {mean_squared_error(average_state_est, list(itertools.chain.from_iterable(real_x)))}\n")
    
    average_state_est_plot(average_state_est, state_std, real_x=real_x, name=name)
    return mseb, average_running_time

def log_mset_plot(files, real_x):
    T = len(real_x)
    labels = []
    log_mses = pd.DataFrame(np.zeros(shape=(T, len(files))))
    for i, f in enumerate(files):
        state_estimates, _, _ = load_hist(f)
        mset = (np.square(state_estimates - real_x.reshape((1, len(real_x))))).mean(axis=0)
        log_mses.iloc[:,i] = np.log(mset)
        labels.append(f.split('/')[-2])
    log_mses.columns = labels

    fig, ax = plt.subplots(figsize=(8, 6))
    lines = ax.plot(log_mses, alpha=0.8)
    ax.legend(lines, labels)
    ax.set(xlabel="T", ylabel="log of MSE",
           title="Trajectories of logarithm of MSE for various algorithms")

@numba.jit
def tail_prob(file, breaks = [7, 30]):
    """
    Analyze the tail probability of a running result (history) file of npz format.

    1. Dsiplay the density plot of the state estimate particles for different limit specifications
    2. TO-DO: Boxplot

    Args:
        file ([type]): [description]
        breaks (list)  : limit specifications of density plot for state estimate particles.
    """
    state_estimates, xs, _ = load_hist(file)
    state_estimates = state_estimates.reshape((state_estimates.size ,))
    xmin, xmax = min(state_estimates), max(state_estimates)
    breaks = sorted(breaks + [xmin, xmax])
    support = np.arange(min(breaks), max(breaks), 0.05)

    # Limited density plot
    kde = sm.nonparametric.KDEUnivariate(state_estimates)
    kde.fit()
    ds = kde.evaluate(support)

    fig, ax = plt.subplots(ncols=2, nrows=int(np.ceil(len(breaks)/2)),
    figsize=(8, 6), constrained_layout=True)
    fig.suptitle(f"Density plot of state estimate particles:{file.split('/')[-1][7:-4]}")
    ax = ax.flatten()
    # sns.kdeplot(state_estimates, clip=(xmin, xmax), ax=ax[0])
    ax[0].plot(support, ds)
    for i in range(len(breaks)-1):
        # sns.kdeplot(state_estimates, clip=(breaks[i], breaks[i+1]), ax=ax[i+1])
        ax[i+1].plot(support, ds)
        ax[i+1].set_xlim(breaks[i], breaks[i+1])
        ymax = max(ds[(breaks[i] < support) & (support < breaks[i+1])])
        ax[i+1].set_ylim(bottom = 0, top=0.01 + ymax*1.2)
    # plt.tight_layout()
    pass

# @numba.jit
def tail_prob_multi(files,
                    real_x,
                    alpha: list =[0.05, 0.1, 0.9, 0.95, 1]
                    # breaks: list =[7,30],
                    ) -> pd.DataFrame:
    """
    Analyze and compare the tail probability of multiple running history files of npz format.

    Args:
        files (list of string): path of few running history files
        real_x (numpy.array): real states
        ~~breaks (list, optional): limit specifications of density plot for state estimate particles. Defaults to [7,30].~~

    Returns:
        pandas.DataFrame: MSE of various quantile estimation.

    Procedure:
        1. Compute the quantiles (for t=1 to t=T) from the samples approximately generated from posterior distribution,
           and use these as standard.
        2. Compute the emperical quantiles from samples generated by various algorithms, the sample's shape is (R, T, N)
           R replications, T time steps, N particles generated in each step
           Estimated quantiles: q_{t, \alpha, alg, rep}
        3. Compute the MSEt for algorithms and quantiles at each time step.
           MSEt: MSE of \alpha-th quantile estimation at time t, MSEt_{t, \alpha, alg}
        4. Taking average over time to get \bar{MSE} for algorithms and quantiles.
           \bar{MSE}_{\alpha, alg}
    """

    # particles_ = []
    # densities = []
    # xmin, xmax = 1000, -1000
    # alpha = [0.05, 0.1, 0.9, 0.95, 1]

    # Get theoretical quantiles from samples generated by BF with large particle size.
    real_quantiles = np.quantile(real_x[1:,:], alpha, axis=1)  # (len(alpha), T)

    logMSEts = []  # log of MSE of quantile estimation for each time step t.
    # $\frac{1}{R} \sum_{r=1}^{R}\left(\widetilde{x}_{t,(\alpha)}^{(r)}-x_{t,(\alpha)}\right)^{2}$
    # R is the number of replications.
    MSEb = []  # Average of MSEs for certain algorithms and quantiles. average over time .
    # $\overline{\mathrm{MSE}}_{\alpha}=\frac{1}{T R} \sum_{t=1}^{T} \sum_{r=1}^{R}\left(\widetilde{x}_{t,(\alpha)}^{(r)}-x_{t,(\alpha)}\right)^{2}$
    labels = []  # labels used in figures

    # Compute quantiles and MSEs
    for f in files:
        state_estimates, xs, _ = load_hist(f)
        # xs: (R, T, N) R-th replication, T-th time step, N-th generated particles
        R, T, _ = xs.shape

        # ====================================================================================================
        # This statement is terriably wrong, the reshape() is not tranpose. The aim is to tranpose the quantiles (3-d array),
        # but reshape make the numbers mixed......
        # quantiles = np.quantile(xs, alpha, axis=2).reshape((R, len(alpha), T))  # quantiles (R, alpha, T)
        # ====================================================================================================
        quantiles = np.quantile(xs, alpha, axis=2).transpose(1,0,2)  # (len(alpha), R, T) => (R, len(alpha), T)
        mset = np.mean(np.square((quantiles[:,:,1:] - real_quantiles)), axis=0)  # square: mse, abs: mae
        logMSEts.append(np.log(mset))
        MSEb.append(np.mean(mset, axis=1))  # Taking average over T
        # quantiles_MSE.append(np.mean(np.square(quantiles - real_quantiles), axis=1))
        state_est = state_estimates.reshape((state_estimates.size ,))
        # particles_.append(state_est)
        labels.append(f.split('/')[-2])
        # xmin, xmax = min(xmin, min(state_est)), max(xmax, max(state_est))
    logMSEts = np.array(logMSEts)

    # Trajectories of logarithm of MSE for various quantiles and algorithms
    fig, ax = plt.subplots(ncols=2, nrows=int(np.ceil(len(alpha)/2)),
    figsize=(8, 3*len(alpha)/2), constrained_layout=True)
    fig.suptitle(f"Trajectories of logarithm of MSE for various quantiles")
    ax = ax.flatten()
    for i in range(len(alpha)):
        lines = ax[i].plot(logMSEts[:,i,:].T, alpha=0.8)
        ax[i].text(0.02, 0.9, str(alpha[i]),
                   size = "large",
                   transform = ax[i].transAxes)
        # ax[i].set_ylim(-4.5, 0)
    fig.legend(lines, labels)
        # if i==0:
        #     ax[i].legend(lines, labels)

    """
    # Density plot of state estimate particles
    breaks = sorted(breaks + [xmin, xmax])
    support = np.arange(min(breaks), max(breaks), 0.05)
    for state_est in particles_:
        kde = sm.nonparametric.KDEUnivariate(state_est)
        kde.fit()
        densities.append(kde.evaluate(support))
    
    fig, ax = plt.subplots(ncols=2, nrows=int(np.ceil(len(breaks)/2)),
    figsize=(8, 6), constrained_layout=True)
    fig.suptitle(f"Density plot of state estimate particles")
    ax = ax.flatten()
    for ds, label in zip(densities, labels):
        ax[0].plot(support, ds, label=label, alpha=0.7)
    ax[0].legend()
    for i in range(len(breaks)-1):
        ymax = 0
        for ds, label in zip(densities, labels):
            ax[i+1].plot(support, ds, label=label, alpha=0.7)
            ax[i+1].set_xlim(breaks[i], breaks[i+1])
            ymax = max(ymax, max(ds[(breaks[i] < support) & (support < breaks[i+1])]))
        ax[i+1].set_ylim(bottom = 0, top=ymax*1.1)
        # ax[i+1].legend()
    """
    logger.info("[FINISH] Tail distribution of {} algorithms' estimated states analyzed.".format(len(files)))
    return pd.DataFrame(MSEb, columns=alpha, index=labels)

def q_dists(files, real_x, alpha: list =[0.05, 0.1, 0.9, 0.95, 1], t=50):
    '''
    To draw the kde plot of quantiles estimated by algorithms and the real quantiles from aid_x, at time t.
    '''
    labels = []
    real_quantiles = np.quantile(real_x[t,:], alpha)
    alg_quantiles = []
    for f in files:
        state_estimates, xs, _ = load_hist(f)
        R, T, _ = xs.shape
        # quantiles = np.quantile(state_estimates, alpha, axis=1)
        quantiles = np.quantile(xs[:,t,:], alpha, axis=1).transpose()
        alg_quantiles.append(quantiles)
        labels.append(f.split('/')[-2])
    alg_quantiles = np.array(alg_quantiles)

    fig, ax = plt.subplots(ncols=2, nrows=int(np.ceil(len(alpha)/2)), figsize=(8, 3*len(alpha)/2), constrained_layout=True)
    ax = ax.flatten()
    # fig.suptitle(f"Trajectories of logarithm of MSE for various quantiles")
    for i in range(len(alpha)):
        for j in range(len(files)):
            sns.kdeplot(alg_quantiles[j][:,i], ax=ax[i], label=labels[j])  # j-th algorithm, all repetitions, i-th quantile
        ax[i].axvline(x=real_quantiles[i])
        ax[i].text(0.02, 0.9, str(alpha[i]),
                   size = "large",
                   transform = ax[i].transAxes)
        if i==0:
            ax[i].legend()
        else:
            ax[i].get_legend().remove()

def q_trajectories(files, real_x, alpha):
    labels = []
    real_quantiles = np.quantile(real_x, alpha, axis=1)
    alg_quantiles = []
    for f in files:
        state_estimates, xs, _ = load_hist(f)
        # xs: (R, T, N) R-th replication, T-th time step, N-th particles generated
        R, T, _ = xs.shape
        # quantiles = np.quantile(state_estimates, alpha, axis=1)
        quantiles = np.quantile(xs, alpha, axis=2).mean(axis=1) # quantiles (R, alpha, T)
        alg_quantiles.append(quantiles)
        labels.append(f.split('/')[-2])
    alg_quantiles = np.array(alg_quantiles)
    fig, ax = plt.subplots(ncols=2, nrows=int(np.ceil(len(alpha)/2)),
                           figsize=(8, 3 * len(alpha) / 2), constrained_layout=True)
    ax = ax.flatten()
    # fig.suptitle(f"Trajectories of logarithm of MSE for various quantiles")
    for i in range(len(alpha)):
        for j in range(len(files)):
            ax[i].plot(alg_quantiles[j][i,:], label=labels[j], alpha=0.8)
            # j-th algorithm, all time steps, average quantile estimation over R replication
        ax[i].plot(real_quantiles[i, :], label='theoretical quantile')
        ax[i].text(0.02, 0.9, str(alpha[i]),
                size="large",
                transform=ax[i].transAxes)
        if i == 0:
            ax[i].legend()



#%% Plot the multivariate observed yields rate
def observation_plot(real_y, tau):
    real_y = np.array(real_y).reshape(100, len(tau))
    T, n = real_y.shape
    fig, ax = plt.subplots(ncols=2, sharey = True, nrows=int(np.ceil(n/2)), figsize=(8, 3*n/2), tight_layout = True, gridspec_kw={"hspace":0})
    ax = ax.flatten()
    for i in range(n):
        ax[i].plot(real_y[:,i], "k-")
        ax[i].text(0.02, 0.9, str(tau[i]),
                   size = "large",
                   transform = ax[i].transAxes)
    fig.text(0.5, -0.02, 'time index', ha='center')


def PXCheck(xs, aid_x, t):
    '''
    To check and compare the distribution of particles at specific time t generated by an algorithm with limited particle
    size but repeated many times and the one with large particle size.
    Draw the KDE plot of the aid_x[T=t], and scatter the xs[T=t]
    '''
    plt.subplots()
    sns.kdeplot(aid_x[t,:], label='aid_x')
    sns.kdeplot(xs[:,t,:].flatten(), label='algorithm')
    plt.legend()

def norm_vs_t(norm, t):
    xs = np.arange(0, 15, 0.1)
    pdf_norm = np.exp(norm.logpdf(xs))
    pdf_t = np.exp(t.logpdf(xs))
    plt.subplots()
    plt.plot(xs, pdf_norm, '-', label='normal')
    plt.plot(xs, pdf_t, '-', label='t')
    plt.legend(loc='best')
    plt.show()
    pass

def dist_cmp(dists, xs=np.arange(0,10,0.01), title=None, ax=None):
    '''
    Draw the density curves of specific distributions, passed in a dictionary.
    :param dists: dictionary of distributions object (particles.distributions.ProbDist type object)
    :param xs: points of x to evaluate the pdf
    '''
    pdfs = np.array([np.exp(d.logpdf(xs)) for d in dists.values()]).T  #.reshape((len(xs), len(dists)))
    if ax != None:
        lines = ax.plot(xs, pdfs, '-')
        ax.legend(lines, list(dists.keys()))
        ax.set(title=title, xlabel='x', ylabel='density')
    else:
        plt.subplots()
        lines = plt.plot(xs, pdfs, '-', ax=ax)
        plt.legend(lines, list(dists.keys()))
        plt.title(title)
        plt.show()

def alg_illustrate_next(alg, aid_x, ax=None):
    alg.next()
    t = alg.t
    xs = alg.hist.X[-1]
    wgts = alg.hist.wgts[-1].W
    s = 20*(wgts - np.min(wgts))/np.ptp(wgts) + 15
    if ax:
        sns.kdeplot(aid_x[t, :], color='black', label='posterior density', ax=ax)
        sns.kdeplot(xs, color='red', label='particles kde', ax=ax)
        ax.scatter(xs, wgts, s=s, alpha=0.8)
    else:
        plt.subplots()
        sns.kdeplot(aid_x[t, :], color='black', label='posterior density')
        sns.kdeplot(xs, color='red', label='particles kde')
        plt.scatter(xs, wgts, s=s, alpha=0.8)
    plt.legend()
    # plt.plot(xs, wgts, '.')
    plt.title(f't={t}')
    pass