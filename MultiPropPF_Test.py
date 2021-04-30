import MultiPropPF
import importlib
import numpy as np
import pandas as pd
from model import CIR
import evaluate
import cProfile
import pstats
import io
import time
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

# record_id = "20210319-012130-2"
record_id = "20210401-170824"
record_id = "_0.5"
loader = np.load(file=f"./Records/CIR{record_id}/data.npz")
real_x, real_y = loader.get("real_x"), loader.get("real_y")

config = CIR.CIR_config(record_id).load()
cir_args = config['MODEL']['PARAMS']
tau, H = config['MODEL']['OBS']['tau'], config['MODEL']['OBS']['H']

loader = np.load(file=f"./Records/CIR{record_id}/aid_x_boot.npz")
aid_x = loader.get('aid_x')
aid_post_mean = loader.get('aid_post_mean')

# tau = np.array([0.25, 0.5, 1, 3, 5, 10])
# H = 0.5

importlib.reload(CIR)


def prof_to_csv(prof: cProfile.Profile, sort='time'):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return '\n'.join(lines)


fk = MultiPropPF.MultiPropFK(ssm=CIR.CIR(tau=tau, H=H), data=real_y,
                             proposals={'normal':0.8, 'normalq':0.1, 'normalql':0.1})
                             # proposals={'t': 0.8, 'normal': 0.1, 'tq': 0.1})
pf = MultiPropPF.MultiPropPF(fk=fk, N=100, ESSrmin=1, resampling="multinomial", store_history=False)

t0 = time.time()
pf.run()
t1 = time.time()
print(f"Computing time: {t1-t0}")

# Profiling ====================================
pr = cProfile.Profile()
pr.enable()
# with PyCallGraph(output=GraphvizOutput()):
pf.run()
pr.disable()
csv = prof_to_csv(pr)
with open("prof2.csv", 'w+') as f:
    f.write(csv)

# after your program ends
pr.print_stats(sort="calls")
# ==============================================

evaluate.resi_plot(pf, aid_post_mean, "MultiPropPF")

plt.subplots()
plt.plot(list(map(lambda i: np.average(pf.hist.X[i], weights=pf.hist.wgts[i].W), range(len(real_x)))))
plt.plot(aid_post_mean)

# ax = sns.distplot(stats.t.rvs(df=5, loc=3.56, scale=2, size=10000), hist=False)
# kde_x, kde_y = ax.lines[0].get_data()
# ax.fill_between(kde_x, kde_y, where=(kde_x >= 0), interpolate=True, color='#388E3C', alpha=0.4)
# ax.fill_between(kde_x, kde_y, where=(kde_x > 8), interpolate=True, color='#EF9A9A', alpha=0.6)
# plt.title("Generate particles around the quantile")
# plt.ylabel("Density")
# plt.xlabel("t distribution")



# Let's compare the algorithms
'''
1. A 'normal' SMC with normal/t proposal only
2. A multiproposal SMC with normal proposal and its two tails, with certain proportions 
3. Bootstrap filter

- Draw the boxplot of particles generated every 10 steps.
- Draw the range of samples

Because the simulation result seems indicate that only 2 more particles from the tail affect the quantiles estimate significantly.
'''

mp_fk = MultiPropPF.MultiPropFK(ssm=CIR.CIR(tau=tau, H=H, tdf=5, ptq=0.95, **cir_args), data=real_y,
                             proposals={'t':0.8, 'tq':0.1, 'tql':0.1})
mp_pf = MultiPropPF.MultiPropPF(fk=mp_fk, N=100, ESSrmin=1, resampling="multinomial", store_history=True)

smc_fk = ssm.GuidedPF(ssm=CIR.CIR(tau=tau, H=H, proposal='t', tdf=5, **cir_args), data=real_y)
smc_pf = particles.SMC(fk=smc_fk, **default_args)

bf_pf = particles.SMC(fk=ssm.Bootstrap(ssm=CIR.CIR(tau=tau, H=H, **cir_args), data=real_y), **default_args)
smc_pf.run()
mp_pf.run()
bf_pf.run()


title = "t  and its tails"
# Box plot of samples
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(10, 6), constrained_layout=True)
fig.suptitle(title)
ax = ax.flatten()
for i, t in enumerate(np.append([1], np.arange(10, 100, 10))):
    xs = pd.DataFrame(np.array([smc_pf.hist.X[t], mp_pf.hist.X[t], bf_pf.hist.X[t]]).T, columns=['smc', 'mppf', 'bf'])
    sns.boxplot(x="variable", y="value", data=pd.melt(xs), ax=ax[i])
    ax[i].set(xlabel='', ylabel='', title=f't={t}')

# Range of samples
xs_range = pd.DataFrame(np.array([np.ptp(np.array(smc_pf.hist.X), axis=1),
                                  np.ptp(np.array(mp_pf.hist.X), axis=1),
                                  np.ptp(np.array(bf_pf.hist.X), axis=1)]).T,
                        columns=['smc', 'mppf', 'bf'])
xs_range.plot()
plt.title(title)