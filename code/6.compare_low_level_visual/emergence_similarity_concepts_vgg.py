import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats
from collections import OrderedDict
import sklearn

zscore = lambda r: (r - np.mean(r, 0, keepdims=True)) / np.std(r, 0, keepdims=True)

##
M = OrderedDict.fromkeys(['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'sem50', 'sem5'])
M['pool1'] = np.load('./data/vgg16/pool1.npy')
M['pool2'] = np.load('./data/vgg16/pool2.npy')
M['pool3'] = np.load('./data/vgg16/pool3.npy')
M['pool4'] = np.load('./data/vgg16/pool4.npy')
M['pool5'] = np.load('./data/vgg16/pool5.npy')
M['sem50'] = np.load( './data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy')
M['sem5'] = np.load( './data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy')[:, :5]

##
nblocks = 390/20
nframes = 25 * 20
R = {'sem50':np.zeros((len(M)-2, nblocks)), 'sem5':np.zeros((len(M)-2, nblocks))}

for iblock in range(nblocks):
    print('block ' + str(iblock))
    st = iblock*nframes
    en = iblock*nframes + nframes if iblock < nblocks-1 else iblock*nframes + nframes -1
    x = np.triu_indices(en-st, 1)
    for i in range(5):
        print(M.keys()[i])
        rsm1 = np.corrcoef(M.values()[i][st:en])[x]
        rsm2 = np.corrcoef(M.values()[-2][st:en])[x]
        R['sem50'][i, iblock] = np.corrcoef(rsm1, rsm2)[0, 1]
        rsm2 = np.corrcoef(M.values()[-1][st:en])[x]
        R['sem5'][i, iblock] =  np.corrcoef(rsm1, rsm2)[0, 1]

##
m = np.mean(R['sem50'], 1)
sem = np.std(R['sem50'], 1)/np.sqrt(nblocks)
fig = plt.figure(figsize=(8, 6), dpi=160)
plt.plot(m, 'k')
plt.fill_between(range(len(m)), m + sem, m - sem, color = 'k', alpha=0.3)

m = np.mean(R['sem5'], 1)
sem = np.std(R['sem5'], 1)/np.sqrt(nblocks)
plt.plot(m, 'grey')
plt.fill_between(range(len(m)), m + sem, m - sem, color='grey', alpha=0.3)
plt.ylim(.3, .8)

##
R = {'sem50':np.zeros((len(M)-2)), 'sem5':np.zeros((len(M)-2))}
x = np.triu_indices(9749, 1)
for i in range(5):
    print(M.keys()[i])
    rsm1 = np.corrcoef(M.values()[i])[x]
    rsm2 = np.corrcoef(M.values()[-2])[x]
    R['sem50'][i] = np.corrcoef(rsm1, rsm2)[0, 1]
    rsm2 = np.corrcoef(M.values()[-1])[x]
    R['sem5'][i] = np.corrcoef(rsm1, rsm2)[0, 1]


## bootstrapping
boot, ci_25, ci_975 = {}, {}, {}
n_samples = 1000
n_boot = 10000
x = np.triu_indices(n_samples, 1)
boot['sem50'] = np.zeros((n_boot, len(M) - 2))
boot['sem5'] = np.zeros((n_boot, len(M) - 2))
ci_25['sem50'] = np.zeros((len(M) - 2))
ci_25['sem5'] = np.zeros((len(M) - 2))
ci_975['sem50'] = np.zeros((len(M) - 2))
ci_975['sem5'] = np.zeros((len(M) - 2))


for i in range(5):
    print(M.keys()[i])

    a1 = M.values()[i]
    a2 = M.values()[-2]
    a3 = M.values()[-1]

    for ib in range(n_boot):
        print(ib)
        frames = sklearn.utils.resample(range(9749), n_samples=n_samples)
        ra1 = a1[frames]
        ra2 = a2[frames]
        ra3 = a3[frames]

        rsm1 = np.corrcoef(ra1)[x]
        rsm2 = np.corrcoef(ra2)[x]
        rsm3 = np.corrcoef(ra3)[x]

        boot['sem50'][ib, i] = np.corrcoef(rsm1, rsm2)[0, 1]
        boot['sem5'][ib, i] = np.corrcoef(rsm1, rsm3)[0, 1]

    ci_25['sem50'][i] = np.percentile(boot['sem50'][:, i], 2.5)
    ci_975['sem50'][i] = np.percentile(boot['sem50'][:, i], 97.5)
    ci_25['sem5'][i] = np.percentile(boot['sem5'][:, i], 2.5)
    ci_975['sem5'][i] = np.percentile(boot['sem5'][:, i], 97.5)


##
fig = plt.figure(figsize=(5, 4), dpi=160)
plt.plot(R['sem50'], 'k')
plt.fill_between(range(len(R['sem50'])), ci_975['sem50'], ci_25['sem50'], color = 'k', alpha=0.3)

plt.plot(R['sem5'], 'grey')
plt.fill_between(range(len(R['sem5'])), ci_975['sem5'], ci_25['sem5'], color = 'grey', alpha=0.3)

plt.ylim(.1, .6)
