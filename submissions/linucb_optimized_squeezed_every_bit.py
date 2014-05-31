#!/usr/bin/env python2.7

import math
import numpy as np
import scipy.linalg

#pylint: disable=C0301,C0111,W0603,W0613
ARTICLE_FEATURES = {}
USER_FEATURES_DIM = 6
LAST_RECOMMENDATION = None
LAST_USER = None
LAS_USER_NORM = None
ALPHA = 0.8

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    for art_key in art:
        ARTICLE_FEATURES[art_key] = {'features' : art[art_key],
                                     'm' : np.identity(USER_FEATURES_DIM),
                                     'b' : np.zeros(USER_FEATURES_DIM),
                                     'w' : np.zeros(USER_FEATURES_DIM),
                                     'updated' : False,
                                     'untouched' : True}

# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    if reward >= 0:
        ARTICLE_FEATURES[LAST_RECOMMENDATION]['m'] += np.outer(LAST_USER,
                                                               LAST_USER)
        ARTICLE_FEATURES['updated'] = True
        ARTICLE_FEATURES['untouched'] = False
        if reward == 1:
            ARTICLE_FEATURES[LAST_RECOMMENDATION]['b'] += LAST_USER

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global LAST_RECOMMENDATION, LAST_USER, LAST_USER_NORM
    LAST_USER = np.array(user_features, copy=False, dtype=np.float64)
    LAST_USER_NORM = np.linalg.norm(LAST_USER)
    LAST_RECOMMENDATION = max(articles, key=calculate_ucb)
    return LAST_RECOMMENDATION

def calculate_ucb(art_id):
    current_article = ARTICLE_FEATURES[art_id]
    if current_article['untouched']:
        ucb = ALPHA*LAST_USER_NORM
        return ucb
    if current_article['updated']:
        current_article['w'] = scipy.linalg.solve(current_article['m'],
                                                  current_article['b'],
                                                  sym_pos=True,
                                                  check_finite=False)
        current_article['updated'] = False
    ucb = np.inner(current_article['w'], LAST_USER) \
        + ALPHA*math.sqrt(np.inner(LAST_USER, scipy.linalg.solve(current_article['m'], LAST_USER,
            sym_post=True, check_finite=False)))
    return ucb
