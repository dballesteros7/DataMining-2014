#!/usr/bin/env python2.7

import sys
import numpy as np
from sklearn.cluster import KMeans


def main():
    data = np.empty((0, 750))
    for line in sys.stdin:
        data = np.vstack((data, np.fromstring(line.strip().split('\t')[1].strip(']').strip('['), sep=',')))
    centroids = KMeans(200).fit(data).cluster_centers_
    for centroid in centroids:
        print str(centroid).strip('[').strip(']').replace(',',' ')
if __name__ == "__main__":
    sys.exit(main())
