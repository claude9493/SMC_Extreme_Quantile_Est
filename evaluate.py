import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error

def resi_plot(alg, real_x, name=""):
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

def average_state_est_plot(average_state_est, real_x, R, name=""):
    plt.figure(figsize=(8,6))
    plt.plot(real_x, color="C0", label="True State")
    plt.plot(average_state_est, linestyle="dashed", color="C0", label=f"{name} State Estimate")
    plt.xlabel(r"time index ($t$)")
    plt.ylabel(r"Average state estimates ($\bar{\hat{x}}_t$)")
    plt.title(f"Average State Estimates from {name} Simulation {R} Repeats")
    plt.legend()

def load_running_result(file):
    temp = np.load(file=file, allow_pickle=True)
    res = temp.get("res")
    state_estimates, running_time = np.stack(res[:,0], axis=0), res[:,1]
    return state_estimates, running_time

def running_result_evaluate(file, real_x, name=""):
    state_estimates, running_time = load_running_result(file)
    R = len(running_time)
    average_running_time = np.mean(running_time)
    average_state_est = np.mean(state_estimates, axis=0)
    print(f"The average running time is {average_running_time}\n\
MSE: {mean_squared_error(average_state_est, list(itertools.chain.from_iterable(real_x)))}")
    average_state_est_plot(average_state_est, real_x, R, name)

def tail_prob_(file):
    pass