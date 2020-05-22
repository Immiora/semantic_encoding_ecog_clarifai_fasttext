import numpy as np
import pickle
import sys
sys.path.insert(0, './code/')
import modeler
from sklearn.linear_model import Ridge, RidgeCV
from modeler.make_custom_cv_vgg import Crossvalidator
from sklearn.metrics import mean_squared_error
import os
from scipy import stats
from scipy.io import savemat
from sklearn.decomposition import PCA

zscore = lambda r: (r - np.mean(r, 0, keepdims=True)) / np.std(r, 0, keepdims=True)

def regress_audio(ttr):
    audio = np.load('./data/audio_envelope_25Hz.npy').astype(np.float32)
    inds = np.load('./results/xcor_37subjs_chill_hfb_audio_100Hz_speech_lags_25Hz_1e-2.npz')['inds']
    lags = np.load('./results/xcor_37subjs_chill_hfb_audio_100Hz_speech_lags_25Hz_1e-2.npz')['lags']
    for counter, ind in enumerate(inds):
        a_ = np.roll(audio, lags[counter])
        ttr[:, ind] = modeler.utils.regress_z(ttr[:, ind], a_)
    return ttr

def regress_boxcar(ttr):
    boxcar = np.hstack([np.zeros(30 * 25), np.ones(30 * 25)] * 7)[:-30 * 25 - 1]
    inds = np.load('./results/ols_37subjs_chill_hfb_boxcar_25Hz_lags_1e-2.npz')['inds']
    lags = np.load('./results/ols_37subjs_chill_hfb_boxcar_25Hz_lags_1e-2.npz')['lags']
    for counter, ind in enumerate(inds):
        b_ = np.roll(boxcar, lags[counter])
        ttr[:, ind] = modeler.utils.regress_z(ttr[:, ind], b_)
    return ttr

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


## run regression
for lag in [320]:
    print('Lag ' + str(lag))
    kfolds = 5
    nfolds = 5
    alphas = [0.1, 1, 10, 1e+3, 1e+4, 3e+4, 5e+4, 1e+5, 1e+7]
    input_var = 'pool2'
    out_dir = './results/encoding/vision_models/ridge_' + input_var + '_pc50_accCV5_alphaCV5/audio_boxcar_regressed_hfb_shift'+str(lag)+'ms/'
    xtr = None
    
    if input_var == 'pxl_color':
        xtr = np.load('./data/chill_pxl_colored.npy').astype(np.float32)
    elif input_var == 'pxl_grey':
        xtr = np.load('./data/chill_pxl_grey.npy').astype(np.float32)
    elif input_var == 'gwp_simple':
        xtr = np.load('./data/chill_gwp_features_simple.npy').astype(np.float32)
    elif input_var == 'gwp_complex':
        xtr = np.load('./data/chill_gwp_features_complex.npy').astype(np.float32)
    elif input_var == 'pool1':
        xtr = np.load('./data/vgg16/pool2.npy').astype(np.float32)
    elif input_var == 'pool2':
        xtr = np.load('./data/vgg16/pool2.npy').astype(np.float32)
    elif input_var == 'pool3':
        xtr = np.load('./data/vgg16/pool3.npy').astype(np.float32)
    elif input_var == 'pool4':
        xtr = np.load('./data/vgg16/pool4.npy').astype(np.float32)
    elif input_var == 'pool5':
        xtr = np.load('./data/vgg16/pool5.npy').astype(np.float32)
            
    y_file = './data/37subjs_25Hz_hfb.npy'
    z_file = './data/audio_envelope_25Hz.npy'
    ttr = np.load(y_file).astype(np.float32)[:-1]
    ztr = np.load(z_file).astype(np.float32)
    xtr = modeler.utils.regress_z(xtr, np.hstack([np.zeros(30 * 25), np.ones(30 * 25)] * 7)[:-30 * 25 - 1]) # block design (boxcar)
    xtr = modeler.utils.regress_z(xtr, ztr) # audio envelope
    ttr = regress_boxcar(ttr)
    ttr = regress_audio(ttr)
    ttr = np.roll(ttr, -lag/40, axis=0) # negative shift for brain as targets: adjust ms to time points

    # do pca
    pca = PCA(random_state=100, n_components=50, whiten=True)
    xtr = pca.fit_transform(xtr.astype(np.float32))

    #
    crossvalidator = Crossvalidator(k=kfolds, make_val=False, shuffle=False)
    Train, Test, indTrain, indTest = crossvalidator(xtr[:,:1], ttr) # memory error for all features, adjusted make_custom_cv for this: make_custom_cv_vgg
    scalers = []

    if (out_dir is not None) & (not os.path.exists(out_dir)): os.makedirs(out_dir)

    #
    for kfold in range(kfolds):
        print('Fold ' + str(kfold))
        scaler = modeler.Preprocessor.Scaler(scale_type='z_score', norm_x=False, norm_t=True, use_sklearn=True)
        ktrain, ktest, _ = scaler([xtr[indTrain[kfold][0].astype(np.int)[:,0]], Train[kfold][1]],
                                  [xtr[indTest[kfold][0].astype(np.int)[:,0]], Test[kfold][1]], None)
        scalers.append(scaler)

        scalers0 = []
        r = []
        for alpha in alphas:
            print('Nested cv: alpha ' + str(alpha))
            crossvalidator = modeler.Preprocessor.Crossvalidator(k=nfolds, make_val=False, shuffle=False, use_sklearn=True)
            nTrain, nTest = crossvalidator(ktrain[0], ktrain[1])
            y_hat = []

            for nfold in range(nfolds):
                print('Nested fold ' + str(nfold))
                scaler0 = modeler.Preprocessor.Scaler(scale_type='z_score', norm_x=False, norm_t=True, use_sklearn=True)
                nktrain, nktest, _ = scaler0(nTrain[nfold], nTest[nfold], None)
                scalers0.append(scaler0)
                nktrain[0] = np.c_[nktrain[0], np.ones(nktrain[0].shape[0])]
                nktest[0] = np.c_[nktest[0], np.ones(nktest[0].shape[0])]

                model0 = Ridge(alpha=alpha, fit_intercept=False)
                model0.fit(nktrain[0], nktrain[1])
                print(mean_squared_error(model0.predict(nktest[0]), nktest[1]))
                y_hat.append(model0.predict(nktest[0]))
            y_hat = np.concatenate(y_hat)
            r.append(np.mean([np.corrcoef(a, b)[0,1] for a, b, in zip(y_hat, ktrain[1])]))
            print('r ' + str(r[-1]))
        r = [-1 if np.isnan(i) else i for i in r]
        alpha_ = alphas[np.argmax(r)]
        print(alpha_)

        ktrain[0] = np.c_[ktrain[0], np.ones(ktrain[0].shape[0])]
        ktest[0] = np.c_[ktest[0], np.ones(ktest[0].shape[0])]
        model = Ridge(alpha=alpha_, fit_intercept=False)
        model.fit(ktrain[0], ktrain[1])

        y_hat = model.predict(ktest[0])
        y = ktest[1]
        print(mean_squared_error(y_hat, y))
        acc = [np.corrcoef(y[:, i], y_hat[:, i])[0,1] for i in range(y.shape[-1])]
        print('Max test accuracy: ' + str(np.max(acc)))
        print(model.coef_.shape)

        # train_hat = model.coef_.dot(ktrain[0].T).T
        np.save(out_dir + '/predictions_fold' + str(kfold), y_hat)
        np.save(out_dir + '/targets_fold' + str(kfold), y)
        pickle.dump(model, open(out_dir + '/model' + str(kfold) + '.p', 'wb'))
        pickle.dump(scalers, open(out_dir + '/scalers' + str(kfold) + '.p', 'wb'))

    print('calculating p-values')
    cr = []
    for kfold in range(kfolds):
        y = np.load(out_dir + 'targets_fold' + str(kfold) + '.npy')
        y_hat = np.load(out_dir + 'predictions_fold' + str(kfold) + '.npy')
        cr.append(arcor(y, y_hat))
    cr = np.array(cr)
    savemat(out_dir + 'accuracy.mat', {'acc': cr})

    alpha = 1e-3
    df = get_df(out_dir)
    n = cr.shape[-1]
    p, pm = get_pval(np.mean(cr, 0), df, alpha / n)
    ##
    savemat(out_dir + 'uncorrected_pvalues.mat', {'p': p})
    savemat(out_dir + 'pmask_mean_over_folds_' + str(alpha) + '_bonf.mat', {'pmask': pm})
    print('done')