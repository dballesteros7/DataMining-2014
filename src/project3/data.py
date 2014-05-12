#!/usr/bin/env python2.7

import numpy as np

# Load the file stored in NPZ format
data = np.load("/home/diegob/workspace/dm-2014/data/project3/arr_0.npz")

# To store the file in CSV format
np.savetxt('/home/diegob/workspace/dm-2014/data/project3/training.csv', data)

# Load CSV file
#data = np.loadtxt('training.csv')
