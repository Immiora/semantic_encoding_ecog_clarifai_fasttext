import numpy as np
import utils
from sklearn.model_selection import KFold
from sklearn import preprocessing
import copy

def check_types():
    pass


class Crossvalidator:
    # currently works only for arrays, along 0 axis
    # save indices to self

    def __init__(self, k, make_val, shuffle, use_sklearn):
        self.k = k
        self.make_val = make_val
        self.shuffle = shuffle
        self.use_sklearn = use_sklearn

    def _shuffle_data(self, x, t):
        idx = np.random.permutation(x.shape[0])
        x = x[idx]
        t = t[idx]
        return x, t

    def _sklearn_kcrossvalidation(self, x, t):

        def _make_folds(x, t, k):
            kf = KFold(n_splits=k)
            Train, Test = [], []
            for i_train, i_test in kf.split(x):
                Train.append([x[i_train], t[i_train]])
                Test.append([x[i_test], t[i_test]])
            return Train, Test

        def _create_val(Train, Test):
            Train_, Val = [], []
            for ktrain, ktest in zip(Train, Test):
                l = ktest[0].shape[0]
                Val.append([ktrain[0][-l:], ktrain[1][-l:]])
                Train_.append([ktrain[0][:-l], ktrain[1][:-l]])
            return Train_, Test, Val

        if self.shuffle: x, t = self._shuffle_data(x, t)

        Train, Test = _make_folds(x, t, self.k)
        if self.make_val:
            return _create_val(Train, Test)
        else:
            return Train, Test


    def _custom_kcrossvalidation(self, x, y):
        '''
        Function creates cross-validation folds for x and y inputs.
        :param x: 3d input array: ntrials x ntimepoints x nfeatures
        :param y: vector of labels: ntrials,
        :param k: number of splits of the data
        :param n_val: number of trials in validation set
        :return: Train, Val and Test lists of nfolds length, each item is a tuple of x and y data
        '''

        n = x.shape[0]
        shape = x.shape[1:]
        x = x.reshape(n, -1)

        if self.shuffle: x, t = self._shuffle_data(x, y)

        l = range(n)
        test_folds = utils.split_list(l, self.k)
        Train, Val, Test = [], [], []

        for i, t in enumerate(test_folds):
            xc = x.copy()
            yc = y.copy()
            Test.append([xc[t].reshape([-1] + [shape[s] for s in range(len(shape))]), y[t]])

            xc = np.delete(xc, t, axis=0)
            yc = np.delete(yc, t, axis=0)

            xc = np.roll(xc, -len(t) * i, axis=0)
            yc = np.roll(yc, -len(t) * i, axis=0)

            Val.append([xc[:len(t)].reshape([-1] + [shape[s] for s in range(len(shape))]), yc[:len(t)]])
            Train.append([xc[len(t):].reshape([-1] + [shape[s] for s in range(len(shape))]), yc[len(t):]])

        return Train, Test, Val

    def __call__(self, x, t):
        #if isinstance(x, list):
        #elif type(x) is np.ndaaray:
        return self._sklearn_kcrossvalidation(x, t) if self.use_sklearn \
                                            else self._custom_kcrossvalidation(x, t)



class Scaler:
    ''' Check/parameter for axis of z-scoring and whether any reshaping is needed'''
    # currently works only for arrays, only 2d, only standard z-scoring (z-score for each column in 2nd dim)
    def __init__(self, scale_type, norm_x, norm_t, use_sklearn):
        self.scale_type = scale_type
        self.norm_x = norm_x
        self.norm_t = norm_t
        self.use_sklearn = use_sklearn

    def _sklearn_scaler(self, ktrain_, ktest_=None, kval_=None):
        # assert dimensionality
        ktrain, ktest, kval = copy.copy(ktrain_), copy.copy(ktest_), copy.copy(kval_)
        self.scalers = {}
        self.scalers['x_scaler'] = {'mean':np.zeros((ktrain[0].shape[-1])), 'std':np.ones((ktrain[0].shape[-1]))}
        self.scalers['t_scaler'] = {'mean': np.zeros((ktrain[1].shape[-1])), 'std': np.ones((ktrain[1].shape[-1]))}

        if self.norm_x:
            x_scaler = preprocessing.StandardScaler()
            ktrain[0] = x_scaler.fit_transform(ktrain[0])
            if ktest is not None: ktest[0]  = x_scaler.transform(ktest[0])
            if kval is not None: kval[0] = x_scaler.transform(kval[0])
            self.scalers['x_scaler']['mean'] = x_scaler.mean_
            self.scalers['x_scaler']['std'] = x_scaler.scale_

        if self.norm_t:
            t_scaler = preprocessing.StandardScaler()
            ktrain[1] = t_scaler.fit_transform(ktrain[1])
            if ktest is not None: ktest[1]  = t_scaler.transform(ktest[1])
            if kval is not None: kval[1] = t_scaler.transform(kval[1])
            self.scalers['t_scaler']['mean'] = t_scaler.mean_
            self.scalers['t_scaler']['std'] = t_scaler.scale_

        return ktrain, ktest, kval

    def _zscore_2dfrom3d(self, x, axis=0, m=None, s=None):

        def z_score(x, axis=0):
            m = np.mean(x, axis=axis, keepdims=True)
            s = np.std(x, axis=axis, keepdims=True)
            return (x - m) / s, m, s

        dims = x.shape
        x0 = x.reshape((-1, dims[-1]))
        if (m is None) & (s is None):
            x0, m, s = z_score(x0, axis=axis)
        else:
            x0 -= m
            x0 /= s

        x0 = x0.reshape(dims)
        return x0, m, s


    def _custom_scaler(self, ktrain, ktest=None, kval=None):
        self.scalers = {}
        self.scalers['x_scaler'] = {'mean':np.zeros((ktrain[0].shape[-1])), 'std':np.ones((ktrain[0].shape[-1]))}
        self.scalers['t_scaler'] = {'mean': np.zeros((ktrain[1].shape[-1])), 'std': np.ones((ktrain[1].shape[-1]))}

        if self.norm_x:
            ktrain[0], m, s = self._zscore_2dfrom3d(ktrain[0])
            print(s)
            if kval is not None:  kval[0] = self._zscore_2dfrom3d(kval[0], 0, m, s)[0]
            if ktest is not None: ktest[0] = self._zscore_2dfrom3d(ktest[0], 0, m, s)[0]
            self.scalers['x_scaler']['mean'] = m
            self.scalers['x_scaler']['std'] = s

        if self.norm_t:
            ktrain[1], m, s = self._zscore_2dfrom3d(ktrain[1])
            if kval is not None:  kval[1] = self._zscore_2dfrom3d(kval[1], 0, m, s)[0]
            if ktest is not None: ktest[1] = self._zscore_2dfrom3d(ktest[1], 0, m, s)[0]
            self.scalers['t_scaler']['mean'] = m
            self.scalers['t_scaler']['std'] = s

        return ktrain, ktest, kval

    def __call__(self, ktrain, ktest=None, kval=None):
        return self._sklearn_scaler(ktrain, ktest, kval) if self.use_sklearn \
                                            else self._custom_scaler(ktrain, ktest, kval)

# optional: with or without sklearn


# assert data types
#
#
# split folds:
#       cv_split_sklearn: train, test, val (different shuffle types)
#       make_kcrossvalidation: custom implementation (different shuffles)
#
#
# check dimensionality
#
#
#
# per fold rescaling:
#       standardize_sklearn: scaler/normalizer, which dimensions
#       zscore_2dfrom3d: modify to scale to range, and which dims