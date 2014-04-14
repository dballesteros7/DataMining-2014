#!/usr/bin/env python2.7

import sys
import numpy as np

_MAPPERS = 10

def main():
    array_sum = np.zeros(100)
    for line in sys.stdin:
        line = line.strip()
        _, weights = line.split('\t')
        weights = np.fromstring(weights, sep = ' ')
        array_sum += weights
    array_sum = array_sum / _MAPPERS
    for coeff in array_sum:
        sys.stdout.write("%f " % coeff)
    sys.stdout.write("\n")

if __name__ == "__main__":
    sys.exit(main())
