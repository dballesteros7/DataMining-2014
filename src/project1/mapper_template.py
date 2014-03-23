#!/usr/bin/env python

import numpy as np
import sys

_hashes = []
_r = 16

def partition(video_id, shingles, line):
    rows = []
    for a,b in _hashes:
        if len(rows) == 16:
            print "%s\t%s" % ("".join(rows), line)
            rows = []
        else:
            rows.append(str(int(np.min(((shingles*a + b) % 10007) % 10000))))

if __name__ == "__main__":
    # Very important. Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=10009)
    for i in xrange(256):
        _hashes.append((np.random.randint(1, 10006), np.random.randint(0, 10006)))
    for line in sys.stdin:
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], sep=" ")
        partition(video_id, shingles, line)
