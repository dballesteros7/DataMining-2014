import sys
import numpy as np
from scipy import spatial

def main():
    centroids = np.loadtxt(sys.argv[1])
    data = np.loadtxt(sys.argv[2])
    distances = spatial.distance.cdist(data, centroids, 'euclidean')
    minimums = np.min(distances, axis=1)
    print sum(np.power(minimums, 2))/data.shape[0]
if __name__ == "__main__":
    sys.exit(main())