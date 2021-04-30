import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def dist_plot(x, title):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(8, 6))

    # Add a graph in each part
    sns.boxplot(x, ax=ax_box)
    sns.distplot(x, ax=ax_hist)
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.title(title)
    plt.text(0.01, 0.8, dist_text(x), transform=plt.gca().transAxes)

def dist_text(x):
    return f"mean = {np.mean(x):.5}\nsd = {np.std(x):.5}\nrange = ({min(x):.5}, {max(x):.5})"

def norm_t_cmp(size=1000, loc=0, scale=1, df=1):
    # Generate random samples
    norm = stats.norm(loc, scale).rvs(size)
    t = stats.t(df, loc, scale).rvs(size)
    dist_plot(norm, f"norm, loc={loc}, scale={scale}")
    plt.show()
    dist_plot(t, f"t, df={df}, loc={loc}, scale={scale}")
    plt.show()

if __name__ == "__main__":
    norm_t_cmp(10000, df=2)
    norm_t_cmp(10000, df=4)