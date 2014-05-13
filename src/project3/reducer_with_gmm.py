#!/usr/bin/env python2.7

import sys
import numpy as np
from sklearn import mixture


def main():
    data = np.empty((0, 750))
    weights = []
    for line in sys.stdin:
        line = line.strip()
        weight = float(line.split('\t')[0])
        weights.append(weight)
        data = np.vstack((data, np.fromstring(line.strip().split('\t')[1].strip(']').strip('['), sep=',')))
    centroids = mixture.GMM(n_components=200).fit(data).means_
    for centroid in centroids:
        print str(list(centroid)).strip('[').strip(']').replace(',', ' ')

if __name__ == "__main__":
    sys.exit(main())
