import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from scipy import stats
from fractions import Fraction
from scipy.signal import resample_poly
from scipy.io import savemat

def resample(x, sr1=25, sr2=125, axis=0):
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)

def cross_correlate(x, y, type='pearson'):
    if type == 'spearman':
        x, y = rankdata(x), rankdata(y)
    x = (x - np.mean(x)) / (np.std(x) * x.shape[0])
    y = (y - np.mean(y)) / np.std(y)
    return np.correlate(x, y, mode='full')

def get_pval(r, df, alpha):
    t = r * np.sqrt(df / (1 - r ** 2))
    p = stats.t.sf(t, df)
    return p, p < alpha

##
sr = 100
x = resample(np.load('./data/37subjs_chill_hfb_125Hz.npy'), sr, 125)
y = resample(np.load('./data/audio_envelope_200Hz.npy'), sr, 200)[:x.shape[0], 0]

##
f = sr * 30
spe = [np.arange(i*f,i*f+f) for i in range(1, 13, 2)]
mus = [np.arange(i*f,i*f+f) for i in [0, 2, 4, 8, 10, 12]]


##
r = np.zeros(x.shape[-1])
for i in range(x.shape[-1]):
    r[i] = np.corrcoef(x[:, i], y)[0, 1]

alpha = 1e-2
df = f - 2
n = r.shape[-1]
p, pm = get_pval(r, df, alpha / n)

##
savemat('./results/xcor_37subjs_chill_hfb_audio_100Hz_1e-2.mat', {'r': r, 'p': p, 'pmask':pm})

########################################### correlation to speech ######################################################
xr = []
for s in spe:
    xr.append([])
    for i in range(x.shape[-1]):
        xr[-1].append(cross_correlate(x[s][:, i], y[s]))
xr = np.array(xr)

##
e = 4
plt.plot(np.mean(xr[:,e, f-2*sr:f+2*sr], 0), 'r')
plt.plot(xr[:,e, f-2*sr:f+2*sr].T, 'r', linewidth=.5, alpha=.3)


##
mxr = np.zeros_like(r)
for i in range(x.shape[-1]):
    mxr[i] = np.max(np.mean(xr[:, i, f:f+sr], 0))

alpha = 1e-2
df = f - 2
n = mxr.shape[-1]
p, pm = get_pval(mxr, df, alpha / n)

##
temp=np.mean(xr[:, np.where(pm==1)[0], f:f+sr], 0)
np.argmax(temp,1)
counts, bins = np.histogram(np.argmax(temp,1), bins=10)
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, counts, align='center', width = 0.7 * (bins[1] - bins[0]))

##
savemat('./results/xcor_37subjs_chill_hfb_audio_50Hz_speech_lags_1e-2.mat', {'r': mxr, 'p': p, 'pmask':pm, 'alpha':alpha, 'lags':np.argmax(temp,1)})

##
labs = np.load('./data/37subjs_elabels.npy')
labs[np.where(pm==1)[0]]
counts, bins = np.histogram(labs[np.where(pm==1)[0]], range(1, 35))
center = (bins[:-1] + bins[1:]) / 2
print(center[counts>0])
print(counts[counts>0])


########################################### correlation to music ######################################################
xr_mus = []
for s in mus:
    xr_mus.append([])
    for i in range(x.shape[-1]):
        xr_mus[-1].append(cross_correlate(x[s][:, i], y[s]))
xr_mus = np.array(xr_mus)

##
e = 1223
plt.plot(np.mean(xr_mus[:,e, f-2*sr:f+2*sr], 0), 'r')
plt.plot(xr_mus[:,e, f-2*sr:f+2*sr].T, 'r', linewidth=.5, alpha=.3)


##
mxr_mus = np.zeros_like(r)
for i in range(x.shape[-1]):
    mxr_mus[i] = np.max(np.mean(xr_mus[:, i, f:f+sr], 0))

alpha = 1e-2
df = f - 2
n = mxr_mus.shape[-1]
p_mus, pm_mus = get_pval(mxr_mus, df, alpha / n)

##
temp=np.mean(xr_mus[:, np.where(pm_mus==1)[0], f:f+sr], 0)
np.argmax(temp,1)
counts, bins = np.histogram(np.argmax(temp,1), bins=10)
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, counts, align='center', width = 0.7 * (bins[1] - bins[0]))

##
savemat('./results/xcor_37subjs_chill_hfb_audio_50Hz_music_lags_1e-2.mat', {'r': mxr_mus, 'p': p_mus, 'pmask':pm_mus, 'alpha':alpha, 'lags':np.argmax(temp,1)})

##
labs[np.where(pm_mus==1)[0]]
counts, bins = np.histogram(labs[np.where(pm_mus==1)[0]], range(1, 35))
center = (bins[:-1] + bins[1:]) / 2
print(center[counts>0])
print(counts[counts>0])


###################################### save all significant ####################################

## adjust lags for 25 Hz
temp = np.mean(xr[:, np.where(pm==1)[0], f:f+sr], 0)
temp_mus = np.mean(xr_mus[:, np.where(pm_mus==1)[0], f:f+sr], 0)
lags = np.round(np.argmax(temp,1)/4.).astype(np.int)
lags_mus = np.round(np.argmax(temp_mus,1)/4.).astype(np.int)

##
inds_all = np.concatenate([np.where(pm==1)[0], np.where(pm_mus==1)[0][np.invert(np.in1d(np.where(pm_mus==1)[0],np.where(pm==1)[0]))]])
lags_all = np.concatenate([lags, lags_mus[np.invert(np.in1d(np.where(pm_mus==1)[0],np.where(pm==1)[0]))]])
mxr_all = np.concatenate([np.max(temp,1), np.max(temp_mus,1)[np.invert(np.in1d(np.where(pm_mus==1)[0],np.where(pm==1)[0]))]])
lags_all = lags_all[np.argsort(inds_all)]
mxr_all = mxr_all[np.argsort(inds_all)]
inds_all = inds_all[np.argsort(inds_all)]

##
np.savez('./results/xcor_37subjs_chill_hfb_audio_100Hz_speech_lags_25Hz_1e-2.npz', r=mxr_all, inds=inds_all, lags=lags_all)
savemat('./results/xcor_37subjs_chill_hfb_audio_100Hz_speech_music_bestlag_25Hz_1e-2.mat', {'r':mxr_all, 'inds':inds_all+1, 'lags':lags_all+1})

