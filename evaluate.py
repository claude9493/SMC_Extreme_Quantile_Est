import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import itertools
from sklearn.metrics import mean_squared_error

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
    plt.plot([state_estimates[i] - real_x[i][0] for i in range(T)], '.')
    plt.hlines(y=0, xmin=0, xmax=T, linestyles="dashed")
    plt.xlabel("time index")
    plt.ylabel(r"$\hat{x}_t - x_t$")
    mse = mean_squared_error(state_estimates, list(itertools.chain.from_iterable(real_x)))
    if name:
        plt.title(f"{name}: Estimated state - Real state (MSE={mse:.3})")
    else:
        plt.title(f"Estimated state - Real state (MSE={mse:.3})")
    # plt.show()

def average_state_est_plot(average_state_est, state_std, real_x, name=""):
    """
    Plot true states and average estimated states from a repeated running algorithm.

    Args:
        average_state_est ([type]): [description]
        real_x ([type]): real_states
        R ([type]): number of repetations
        name (str, optional): [description]. Defaults to "".
    
    Refer to Figure 2 of Neslihanoglu, S., & Date, P. (2019).
    Neslihanoglu, S., & Date, P. (2019). A modified sequential Monte Carlo procedure for the efficient recursive estimation of extreme quantiles. Journal of Forecasting, 38(5), 390-399.
    """
    _, ax = plt.subplots(figsize = (8,6))
    T = len(real_x)
    plt.plot(real_x, color="C0", label="True State")
    plt.plot(average_state_est, linestyle="dashed", color="C0", label=f"{name} State Estimate")

    lower = average_state_est - state_std * 1.96
    upper = average_state_est + state_std * 1.96
    plt.fill_between(np.linspace(0, T, T), lower, upper, alpha=0.2)
    
    ax.set(title = f"Average State Estimates from {name} Simulation",
        xlabel = r"time index ($t$)",
        ylabel = r"Average state estimates ($\bar{\hat{x}}_t$)")

    plt.legend()

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
        print(f"[INFO] History loaded\nAlgorithm: {meta['alg_name']}\nCreated time: {meta['timestamp']}\n\n")

    res = loader.get("res")
    state_estimates, running_time = np.stack(res[:,0], axis=0), res[:,1]
    return state_estimates, running_time

def result_evaluate(file, real_x, name=""):
    """
    Evaluate running result, calculate the average running time of R repeatitions, mean squared error of the average state estimates, and plot the average state estimates with real states.

    Args:
        file (string): path of the result file.
        real_x (numpy.array): real_states
        name (str, optional): description of the algorithm. Defaults to "".
    """
    state_estimates, running_time = load_hist(file)
    # R = len(running_time)
    average_running_time = np.mean(running_time)
    average_state_est = np.mean(state_estimates, axis=0)
    state_std = np.std(state_estimates, axis=0)
    print(f"The average running time is {average_running_time}\n\
MSE: {mean_squared_error(average_state_est, list(itertools.chain.from_iterable(real_x)))}")
    
    average_state_est_plot(average_state_est, state_std, real_x, name)

def tail_prob(file, breaks = [7, 30]):
    """
    Analyze the tail probability of a running result (history) file of npz format.

    1. Dsiplay the density plot of the state estimate particles for different limit specifications
    2. TO-DO: Boxplot

    Args:
        file ([type]): [description]
        breaks (list)  : limit specifications of density plot for state estimate particles.
    """
    state_estimates, _ = load_hist(file)
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

def tail_prob_multi(files, real_x, breaks=[7,30]):
    """
    Analyze and compare the tail probability of multiple running history files of npz format.

    Args:
        files (list of string): path of few running history files
        real_x (numpy.array): real states
        breaks (list, optional): limit specifications of density plot for state estimate particles. Defaults to [7,30].

    Returns:
        pandas.DataFrame: MSE of various quantile estimation.
    """

    particles = []
    labels = []
    densities = []
    xmin, xmax = 1000, -1000
    qs = [0.05, 0.1, 0.9, 0.95, 1]
    real_quantiles = np.quantile(real_x, qs).reshape((5,1))
    quantiles_MSE = []
    for f in files:
        temp, _ = load_hist(f)
        quantiles = np.quantile(temp, qs, axis=1)
        quantiles_MSE.append(np.mean((quantiles - real_quantiles)**2, axis=1))
        state_est = temp.reshape((temp.size ,))
        particles.append(state_est)
        labels.append(f.split('/')[-2])
        
        xmin, xmax = min(xmin, min(state_est)), max(xmax, max(state_est))
    
    # sns.boxplot(data=particles)
    # plt.legend(labels)
    # plt.show()

    breaks = sorted(breaks + [xmin, xmax])
    support = np.arange(min(breaks), max(breaks), 0.05)
    for state_est in particles:
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
    return pd.DataFrame(quantiles_MSE, columns=qs, index=labels)