#!/usr/bin/env python2.7

import sys
from sklearn.cluster import KMeans
import numpy as np


def main():
    data = np.empty((0, 750))
    weights = []
    for line in sys.stdin:
        line = line.strip()
        weight = float(line.split('\t')[0])
        weights.append(weight)
        data = np.vstack((data, np.fromstring(line.strip().split('\t')[1].strip(']').strip('['), sep=',')))
    centroids = KMeans(200).fit(data).cluster_centers_
    for centroid in centroids:
        print str(list(centroid)).strip('[').strip(']').replace(',', ' ')

if __name__ == "__main__":
    sys.exit(main())
