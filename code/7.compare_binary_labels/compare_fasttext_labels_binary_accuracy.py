import numpy as np
from collections import OrderedDict
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from wilcoxon import wilcoxon
import pandas as pd
import sklearn
import matplotlib.ticker as ticker

zscore = lambda r: (r - np.mean(r, 0, keepdims=True)) / np.std(r, 0, keepdims=True)

##
R = OrderedDict.fromkeys(['lab', 'sem'])
P = OrderedDict.fromkeys(['lab', 'sem'])

R['lab'] = loadmat('./results/_encoding/ridge_accCV5_alphaCV5_concepts_129_binary/audio_boxcar_regressed_brain_hfb_shift320ms/accuracy.mat')['acc']
R['sem'] = loadmat('./results/_encoding/ridge_semantic_pc50_concepts129_accCV5_alphaCV5/audio_boxcar_regressed_hfb_shift320ms/accuracy.mat')['acc']

P['lab'] = loadmat('./results/_encoding/ridge_accCV5_alphaCV5_concepts_129_binary/audio_boxcar_regressed_brain_hfb_shift320ms/uncorrected_pvalues.mat')['p']
P['sem'] = loadmat('./results/_encoding/ridge_semantic_pc50_concepts129_accCV5_alphaCV5/audio_boxcar_regressed_hfb_shift320ms/uncorrected_pvalues.mat')['p']


for k in R.keys():
    R[k] = np.mean(R[k], 0)
    P[k] = P[k].flatten()

################################# difference in accuracy with selected baseline ########################################

k = 'lab'
alpha = 1e-3
n = len(R['sem'])
ind1 = np.where(P[k] < alpha / n)[0]
ind2 = np.where(P['sem'] < alpha / n)[0]
inds = np.union1d(ind1, ind2)

# median for the difference
mask = R['sem'][inds] - R[k][inds]>.1
np.median((R['sem'][inds] - R[k][inds])[mask])

# descriptive stats
print('median ' + k + ': ', np.median(R[k][inds]))
print('median sem: ', np.median(R['sem'][inds]))
print('range acc ' + k + ': ', np.min(R[k][ind1]), np.max(R[k][ind1]))
print('range acc sem: ', np.min(R['sem'][ind2]), np.max(R['sem'][ind2]))
print('neles ' + k + ': ', len(ind1))
print('neles sem: ', len(ind2))
print('% improve: ', np.sum(R['sem'][inds] - R[k][inds]>.1) / (len(inds) / 100.))

print(wilcoxon(R['sem'][inds], R[k][inds], alternative='greater'))


## scatter plots distribution over all electrodes
sns.set_context("talk")
sns.set(style="ticks")

g = sns.jointplot(x=R[k], y=R['sem'], kind="reg",
                  xlim=(-.15, .55), ylim=(-.15, .55), height = 4, color='k',
                  marginal_kws={'bins':100, 'hist':True, 'kde':True},scatter_kws={'s':50})
for i,j in zip(R[k], R['sem']):
    if j - i > .1:
        g.ax_joint.plot(i, j, markerfacecolor=(.8, .1, .1, .7), marker='o', markeredgecolor=(.5, .1, .1, 1), markersize=8)
    elif i - j > .1:
        g.ax_joint.plot(i, j, markerfacecolor=(.1, .1, .8, .7), marker='o', markeredgecolor=(.1, .1, .5, 1), markersize=8)
g.ax_joint.axhline(y=np.median(R['sem'][inds]), color=(.8, .1, .1, .7), linewidth = 2)
g.ax_joint.axvline(x=np.median(R[k][inds]), color=(.1, .1, .8, .7), linewidth = 2)
g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(.1))
g.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(.1))


## scatter plots only significantly fitted electrodes: ind2
sns.set_context("talk")
sns.set(style="darkgrid")
g = sns.JointGrid(x = R[k][inds], y = R['sem'][inds], height = 4,  xlim=(-.15, .55), ylim=(-.15, .55))
g.plot_joint(plt.scatter, color='k')
g.plot_joint(sns.regplot, color='k')
g.plot_marginals(sns.distplot, color='k')
for i,j in zip(R[k][inds], R['sem'][inds]):
    if j - i > .1:
        g.ax_joint.plot(i, j, markerfacecolor=(.8, .1, .1, .7), marker='o', markeredgecolor=(.5, .1, .1, 1), markersize=8)
    elif i - j > .1:
        g.ax_joint.plot(i, j, markerfacecolor=(.1, .1, .8, .7), marker='o', markeredgecolor=(.1, .1, .5, 1), markersize=8)

g.ax_joint.axhline(y=np.median(R['sem'][inds]), color=(.8, .1, .1, .7), linewidth = 2)
g.ax_joint.axvline(x=np.median(R[k][inds]), color=(.1, .1, .8, .7), linewidth = 2)
g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(.1))
g.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(.1))



########################################## difference in accuracy per ROI ##############################################

labels = np.load('./data/37subjs_elabels.npy')
with open('./data/electrode_label_list.txt', 'r') as fid: labnames = fid.readlines()
labnames = map(lambda x: x.split('.')[0], labnames)


labnames_ = [labnames[i-1] for i in np.unique(labels)]
D = {'roi':[], 'r_dif':[]}
temp = labels[inds]
means = {}

for iroi, roi in enumerate(np.unique(labels)):
    a1_, a2_, = R['sem'][inds][temp == roi], R[k][inds][temp == roi]
    b1_, b2_, = P['sem'][inds][temp == roi], P[k][inds][temp == roi]

    if np.sum(temp == roi) > 2:
        print(iroi, roi, labnames_[iroi], len(np.where(b1_ < alpha / n)[0]), len(np.where(b2_ < alpha / n)[0]),
              len(a1_), np.round(np.mean(a1_), 3), np.round(np.mean(a2_), 3),
              wilcoxon(a1_, a2_, alternative='two-sided'))
        D['roi'].append([labnames_[iroi]]*len(a1_))
        D['r_dif'].append(a1_ - a2_)
        means[labnames_[iroi]] = np.mean(a1_) - np.mean(a2_)

for key in D.keys():
    D[key] = [item for sublist in D[key] for item in sublist]
D = pd.DataFrame(D)


## bootstraping
boot, ci_25, ci_975 = {}, {}, {}

for iroi, roi in enumerate(np.unique(labels)):
    a1_, a2_, = R['sem'][inds][temp == roi], R[k][inds][temp == roi]

    if np.sum(temp == roi) > 2:
        boot[labnames_[iroi]] = np.zeros((1000))

        for ib in range(1000):
            boot[labnames_[iroi]][ib] = np.median(sklearn.utils.resample(a1_)) - np.median(sklearn.utils.resample(a2_))
        ci_25[labnames_[iroi]] = np.percentile(boot[labnames_[iroi]], 2.5)
        ci_975[labnames_[iroi]] = np.percentile(boot[labnames_[iroi]], 97.5)

##
temp =  [ci_25.keys()[i] for i in np.where(np.array(ci_25.values())>.01)[0]]
order = [temp[i] for i in np.argsort([means[key] for key in temp])]
plt.figure(figsize=(6, 3))
sns.set_context("paper")
sns.barplot(x='roi', y='r_dif', data=D[D['roi'].isin(temp)], order=order)