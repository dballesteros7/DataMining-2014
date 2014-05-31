#!/usr/bin/env python2.7

import numpy as np
import sys
import math as math

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    global _alpha
    global _all_art
    global _M
    global _b
    global _rec_art
    global _curr_x

    _alpha = 1
    _all_art = {}
    _M = {}
    _b = {}
    _rec_art = 0
    _curr_x = np.zeros(36)
    
    _all_art = art
    for i in art:
        _M[i] = np.identity(36)
        _b[i] = np.zeros(36)
    


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    if reward >= 0:
        for i in _M:
            _M[i] = _M[i]+_curr_x.dot(np.transpose(_curr_x))
            _b[i] = _b[i]+reward*_curr_x

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    w = {}
    ucb  = {}
    max_ucb = 0
    for i in articles:
        x = np.outer(_all_art[i], user_features)
        x = x.flatten()
        w[i] = np.linalg.inv(_M[i]).dot(_b[i])
        ucb[i] = np.transpose(w[i]).dot(x)
        ucb[i] += _alpha*math.sqrt(np.transpose(x).dot(np.linalg.inv(_M[i])).dot(x))
        if ucb[i]>max_ucb:
            max_ucb = ucb[i]
            _rec_art = i
            _curr_x = x
    return _rec_art