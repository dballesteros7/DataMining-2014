#!/usr/bin/env python

import numpy as np
import sys


def are_they_85_percent_similar(videoA, videoB):
    sim = len(videoA[1] & videoB[1])/float(len(videoA[1] | videoB[1]))
    if sim >= 0.85:
        return True
    return False

def produce_duplicates(video_infos):
    processed_videos = [(int(x[6:15]), set(np.fromstring(x[16:], sep=" "))) for x in video_infos]
    for idx, video_base in enumerate(processed_videos):
        for video_compare in processed_videos[idx + 1:]:
            if are_they_85_percent_similar(video_base, video_compare):
                print "%d\t%d" % (min(video_base[0], video_compare[0]),
                                  max(video_base[0], video_compare[0]))

last_key = None
key_count = 0
candidates = set()

for line in sys.stdin:
    line = line.strip()
    key, video_info = line.split("\t")

    if last_key is None:
        last_key = key

    if key == last_key:
        candidates.add(video_info)
    else:
        # Key changed (previous line was k=x, this line is k=y)
        produce_duplicates(candidates)
        candidates = set()
        candidates.add(video_info)
        last_key = key

if len(candidates) > 0:
    produce_duplicates(candidates)
