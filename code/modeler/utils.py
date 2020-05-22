import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
import chainer.functions as F
import chainer.links as L

def split_list(l, wanted_parts=1):
    length = len(l)
    return [ l[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def shuffle_list(l):
    ind = np.random.permutation(range(len(l)))
    return [l[i] for i in ind]


def trim(x, n):
    n_trim = x.shape[0] - x.shape[0]/n * n
    return x[:-n_trim] if n_trim > 0 else x


def roll(x, n_roll=0):
    return np.roll(x, shift=n_roll, axis=0)


def reshape3(x, dim2):
    return x.reshape(x.shape[0]/dim2, dim2, x.shape[-1])


def regress_z(x, z):
    c = np.linalg.lstsq(np.c_[z, np.ones(z.shape[0])], x)[0]
    r = x - np.c_[z, np.ones(z.shape[0])].dot(c)
    return r.astype(np.float32)


def resample(x, sr1=25, sr2=125, axis=0):
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)

def extend_dim_conv(x):
    return F.expand_dims(x, 1)