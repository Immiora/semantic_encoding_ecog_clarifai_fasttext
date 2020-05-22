import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats
from collections import OrderedDict
import sys
sys.path.insert(0, './code/')
from wilcoxon import wilcoxon
from scipy.stats import ttest_rel


zscore = lambda r: (r - np.mean(r, 0, keepdims=True)) / np.std(r, 0, keepdims=True)

##
M = OrderedDict.fromkeys(['pxl', 'gwp', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'sem50', 'sem5'])
M['pxl'] = np.load('./data/chill_pxl_colored.npy')
M['gwp'] = np.load('./data/chill_gwp_features_complex.npy')
M['pool1'] = np.load('./data/vgg16/pool1.npy')
M['pool2'] = np.load('./data/vgg16/pool2.npy')
M['pool3'] = np.load('./data/vgg16/pool3.npy')
M['pool4'] = np.load('./data/vgg16/pool4.npy')
M['pool5'] = np.load('./data/vgg16/pool5.npy')
M['sem50'] = np.load( './data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy')
M['sem5'] = np.load( './data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy')[:, :5]

##
for k in M.keys():
    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax1 = fig.add_subplot(111)
    img = ax1.imshow(1-np.corrcoef(M[k]))
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cb = plt.colorbar(img, cax=cbaxes)

##
rs = np.zeros((len(M), len(M)))
x = np.triu_indices(9749, 1)
for i, j in zip(*np.triu_indices(len(M))):
    print(i, j)
    if i != j:
        rdm1 = 1 - np.corrcoef(M.values()[i])[x]
        rdm2 = 1 - np.corrcoef(M.values()[j])[x]
        rs[i, j] = 1 - np.corrcoef(rdm1, rdm2)[0, 1]
        rs[j, i] = rs[i, j]

##
fig = plt.figure(figsize=(8, 6), dpi=160)
ax1 = fig.add_subplot(111)
img = ax1.imshow(rs)
plt.yticks(range(len(M)), M.keys())
plt.xticks(range(len(M)), M.keys())
cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
cb = plt.colorbar(img, cax=cbaxes)

##
rdm1 = 1 - np.corrcoef(M['gwp'])[x]
rdm2 = 1 - np.corrcoef(M['sem50'])[x]

print(wilcoxon(rdm1, rdm2, alternative='two-sided'))
print(ttest_rel(rdm1, rdm2))