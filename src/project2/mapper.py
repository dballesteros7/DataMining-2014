#!/usr/bin/env python2.7

import sys

from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.kernel_approximation import RBFSampler

import numpy as np

_LOSS = 'hinge'
_PENALTY = 'l1'
_REGULARIZATION=0.0001
_KEY = 1
_GAMMA = 10
_M = 100


# This function has to either stay in this form or implement the
# feature mapping. For details refer to the handout pdf.
def transform(x_original):
    sampler = RBFSampler(_GAMMA, _M, 10009)
    x_new = sampler.fit_transform(x_original)
    return x_new

def main():
    raw_data = np.loadtxt(sys.stdin)
    samples = raw_data.shape[0]
    X = np.empty((samples, _M))
    Y = raw_data[:,0]
    for i in xrange(samples):
        X[i] = transform(raw_data[i, 1:])
    clf = SGDClassifier(loss = _LOSS, penalty = _PENALTY,
                        fit_intercept = False, shuffle = True,
                        alpha = _REGULARIZATION)
    clf.fit(X, Y)
    sys.stdout.write('%s\t' % _KEY)
    for coeff in clf.coef_.flatten():
        sys.stdout.write("%f " % coeff)
    sys.stdout.write("\n")

if __name__ == "__main__":
    sys.exit(main())
