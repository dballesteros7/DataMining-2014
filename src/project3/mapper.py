#!/usr/bin/env python2.7

import math
import sys

from scipy import spatial

import numpy as np


def main():
    #print "Starting"
    raw_data = np.loadtxt(sys.stdin)
    original_data = np.copy(raw_data)
    #print "Data loaded..."
    uniform_sampled = np.empty((0,750))
    while(raw_data.shape[0] >= 600):
        sample = np.random.choice(raw_data.shape[0], 600, replace=False)
        data_sample = raw_data[sample, :]
        distances = spatial.distance.cdist(raw_data, data_sample, 'euclidean')
        min_distances = np.min(distances, axis=1)
        sorted_indexes =  np.argsort(min_distances)
        raw_data = np.delete(raw_data, sorted_indexes[:math.ceil(raw_data.shape[0]/2)], axis=0)
        uniform_sampled = np.vstack((uniform_sampled, data_sample))
    voronoi = spatial.distance.cdist(original_data, uniform_sampled, 'euclidean')
    cell_index = np.argmin(voronoi, axis=1)
    cell_map = {}
    for i in cell_index:
        if i not in cell_map:
            cell_map[i] = 0
        cell_map[i] += 1
    #print "Data sampled and voronoi distances calculated..."
    weights = np.empty((original_data.shape[0],))
    accum_dist = 0.0
    for index in xrange(original_data.shape[0]):
        accum_dist += voronoi[index][cell_index[index]]**2
    for index in xrange(original_data.shape[0]):
        current_distance = voronoi[index][cell_index[index]]**2
        weights[index] = math.ceil(5.0 / cell_map[cell_index[index]] + current_distance/(accum_dist - current_distance))
    weight_sum = np.sum(weights)
    probabilities = np.empty(weights.shape)
    for index in xrange(weights.shape[0]):
        probabilities[index] = weights[index]/weight_sum
    final_sampling = np.random.choice(original_data.shape[0], size=20, p=probabilities,
                                      replace=False)
    #final_sample = original_data[final_sampling]
    for index in final_sampling:
        print "%s\t%s" % (1/(20*weights[index]), list(original_data[index,:]))
    return 0
if __name__ == "__main__":
    sys.exit(main())
