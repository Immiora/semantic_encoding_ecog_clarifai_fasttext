import numpy as np
import pickle
import sys
sys.path.insert(0, './code/')
import modeler
from sklearn.linear_model import Ridge
from modeler.make_custom_cv import Crossvalidator
import os
from scipy import stats
import statsmodels.api as sm

##
zscore = lambda r: (r - np.mean(r, 0, keepdims=True)) / np.std(r, 0, keepdims=True)

def arcor(x, y):
    # n_channels is dim 1
    # column-wise correlation for x (n x m) and y (n x m), output is z (m x 1)
    return np.array(map(lambda x_, y_: np.corrcoef(x_, y_)[0, 1], x.T, y.T))

def get_df(load_dir):
    y = np.load(load_dir + 'targets_fold0.npy')
    n = y.shape[0]
    return n-2

def get_pval(r, df, alpha):
    t = r * np.sqrt(df / (1 - r ** 2))
    p = stats.t.sf(t, df)
    return p, p < alpha


## load data
kfolds = 5
nfolds = 5
alphas = [0.1, 1, 10, 1e+3, 1e+4, 3e+4, 5e+4, 1e+5, 1e+7]
out_dir = './results/encoding/ridge_binary_clarifai_concepts2semantic_pcs50_accCV5_alphaCV5/noaudio_noboxcar_regressed/'
x_file = './data/corrected_clarifai_concepts_129_binary.npy'
y_file = './data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy'
xtr = np.load(x_file).astype(np.float32)
ttr = np.load(y_file).astype(np.float32)

##
crossvalidator = Crossvalidator(k=kfolds, make_val=False, shuffle=False)
Train, Test = crossvalidator(xtr, ttr)
scalers = []

if (out_dir is not None) & (not os.path.exists(out_dir)): os.makedirs(out_dir)

##
for kfold in range(kfolds):
    print('Fold ' + str(kfold))
    scaler = modeler.Preprocessor.Scaler(scale_type='z_score', norm_x=False, norm_t=False, use_sklearn=True)
    ktrain, ktest, _ = scaler(Train[kfold], Test[kfold], None)
    scalers.append(scaler)

    scalers0 = []
    r = []
    for alpha in alphas:
        #print('Nested cv: alpha ' + str(alpha))
        crossvalidator = modeler.Preprocessor.Crossvalidator(k=nfolds, make_val=False, shuffle=False, use_sklearn=True)
        nTrain, nTest = crossvalidator(ktrain[0], ktrain[1])
        y_hat = []

        for nfold in range(nfolds):
            # print('Nested fold ' + str(nfold))
            scaler0 = modeler.Preprocessor.Scaler(scale_type='z_score', norm_x=False, norm_t=False, use_sklearn=True)
            nktrain, nktest, _ = scaler0(nTrain[nfold], nTest[nfold], None)
            scalers0.append(scaler0)

            model0 = Ridge(alpha=alpha)
            model0.fit(nktrain[0], nktrain[1])
            y_hat.append(model0.predict(nktest[0]))
        y_hat = np.concatenate(y_hat)
        r.append(np.mean([np.corrcoef(a, b)[0,1] for a, b, in zip(y_hat, ktrain[1])]))
        # print('r ' + str(r[-1]))
    r = [-1 if np.isnan(i) else i for i in r]
    alpha_ = alphas[np.argmax(r)]

    model = Ridge(alpha=alpha_)
    model.fit(ktrain[0], ktrain[1])

    y_hat = model.predict(ktest[0])
    y = ktest[1]
    acc = [np.corrcoef(y[:, i], y_hat[:, i])[0,1] for i in range(y.shape[-1])]
    print('Max test accuracy (pearson r): ' + str(np.max(acc)))
    hat = ktrain[0].dot(np.linalg.pinv(ktrain[0].T.dot(ktrain[0]) + model.alpha*np.eye(ktrain[0].shape[-1]))).dot(ktrain[0].T)
    enp = np.trace(2*hat - hat.dot(hat.T)) # effective number of parameters
    dfe = y.shape[0] - enp
    dfm = enp - 1
    mse = np.mean(np.sum((y - y_hat) ** 2, 0) / dfe)
    msm = np.mean(np.sum((y_hat - np.mean(y, 0)) ** 2, 0) / dfm)
    print('Effective number of parameters: ' + str(enp) + '/' + str(ktrain[0].shape[-1]))
    print('MSE: ' + str(mse)) # mean squares error
    print('MSM: ' + str(msm)) # mean squares model
    print('F stat: ' + str(msm / mse))
    print('R2: ' + str(model.score(ktest[0], y)))
    print('Weight matrix size: ' + str(model.coef_.shape))

    np.save(out_dir + '/predictions_fold' + str(kfold), y_hat)
    np.save(out_dir + '/targets_fold' + str(kfold), y)
    pickle.dump(model, open(out_dir + '/model' + str(kfold) + '.p', 'wb'))
    pickle.dump(scalers, open(out_dir + '/scalers' + str(kfold) + '.p', 'wb'))

##
sm_f, sm_fp, sm_ts, sm_tps, sm_w = [], [], [], [], []
for c in range(6):
    est = sm.OLS(ttr[:,c], xtr)
    est2 = est.fit()
    sm_f.append(est2.fvalue)
    sm_fp.append(est2.f_pvalue)
    sm_ts.append(est2.tvalues)
    sm_tps.append(est2.pvalues)
    sm_w.append(est2.params)
    print(c)
    print(est2.summary())
sm_f, sm_fp, sm_ts, sm_tps, sm_w  = np.array(sm_f), np.array(sm_fp), np.array(sm_ts), np.array(sm_tps), np.array(sm_w)


##
weights = []
for kfold in range(kfolds):
    model_recon = pickle.load(open(out_dir + 'model'+str(kfold)+'.p', 'rb'))
    weights.append(model_recon.coef_)

weights = np.mean(np.array(weights), axis=0)

##
concepts = np.load('./data/corrected_clarifai_concepts_129_binary.npz')['names']

##
pc = 1
print('Min extreme weights: ')
print([concepts[i] for i in np.argsort(weights[pc])[:20]])
print(np.sort(weights[pc])[:20])


print('Max extreme weights: ')
print([concepts[i] for i in np.argsort(weights[pc])[-20:][::-1]])
print(np.sort(weights[pc])[-20:][::-1])
