import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import statsmodels.api as sm

zscore = lambda x: (x - np.mean(x, 0, keepdims=True)) / np.std(x, 0, keepdims=True)

## load data, create boxcar regressor
x = np.load('./data/37subjs_25Hz_hfb.npy')
y = np.hstack([np.zeros(30 * 25), np.ones(30 * 25)] * 7)[:-30 * 25]
x = zscore(x)

## run OLS
sm_f, sm_fp, sm_ts, sm_tps, sm_w = [], [], [], [], []
for e in range(x.shape[-1]):
    sm_f.append([])
    sm_fp.append([])
    sm_ts.append([])
    sm_tps.append([])
    sm_w.append([])
    for i in range(25):
        y_ = np.roll(y, i)
        est = sm.OLS(x[:, e], y_)
        est2 = est.fit()
        sm_f[-1].append(est2.fvalue)
        sm_fp[-1].append(est2.f_pvalue)
        sm_ts[-1].append(est2.tvalues)
        sm_tps[-1].append(est2.pvalues)
        sm_w[-1].append(est2.params)
sm_f, sm_fp, sm_ts, sm_tps, sm_w  = np.array(sm_f), np.array(sm_fp), np.array(sm_ts), np.array(sm_tps), np.array(sm_w)

##
sm_ts = sm_ts[:,:,0]
sm_tps = sm_tps[:,:,0]

##
ts = np.max(sm_ts,1)
lags = np.argmax(sm_ts,1)
ps = np.array([sm_tps[counter, i] for counter, i in enumerate(lags)])

## threshold pvalues and select lags for the electrodes with a significant fit
alpha = 1e-2
inds1 = np.where(ps < alpha/x.shape[-1])[0]
inds2 = np.where(ts > 0)[0]
inds = np.intersect1d(inds1, inds2)
lags_ = lags[inds]

## distribution of lags over rois
labs = np.load('./data/37subjs_elabels.npy')
counts, bins = np.histogram(labs[inds], range(1, 35))
center = (bins[:-1] + bins[1:]) / 2
print(center[counts>0])
print(counts[counts>0])

## save results
np.savez('./results/ols_37subjs_chill_hfb_boxcar_25Hz_lags_1e-2.npz', t=ts[inds], inds=inds, lags=lags_)
savemat('./results/ols_37subjs_chill_hfb_boxcar_25Hz_bestlag_1e-2.mat', {'t':ts[inds], 'inds':inds+1, 'lags':lags_+1})