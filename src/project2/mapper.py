#!/usr/bin/env python2.7

import sys

from sklearn.linear_model.stochastic_gradient import SGDClassifier

import numpy as np

_LOSS = 'hinge'
_PENALTY = 'l1'
_REGULARIZATION=0.00001
_KEY = 1
_M = 5000
_D = 400

_I = np.identity(_D)
_ZEROS = np.zeros(_D)

np.random.seed(10009)
b = np.random.uniform(0, 2*np.pi, _M)
w = np.random.multivariate_normal(_ZEROS, _I, _M)

# This function has to either stay in this form or implement the
# feature mapping. For details refer to the handout pdf.
def transform(x_original):
    x_new = np.empty(_M)
    for i in xrange(_M):
        x_new[i] = np.sqrt(2.0/_M) * np.cos(np.inner(w[i, :], x_original) + b[i])
    #return np.hstack((x_original, [1]))
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
