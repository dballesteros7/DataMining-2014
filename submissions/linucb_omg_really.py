#!/usr/bin/env python2.7

import math
import numpy as np
from scipy import linalg

#pylint: disable=C0301,C0111,W0603,W0613
ARTICLE_FEATURES = {}
USER_FEATURES_DIM = 6
LAST_RECOMMENDATION = None
LAST_USER = None
LAS_USER_NORM = None
ALPHA = 5

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    for art_key in art:
        ARTICLE_FEATURES[art_key] = {'features' : art[art_key],
                                     'm' : np.identity(USER_FEATURES_DIM),
                                     'b' : np.zeros(USER_FEATURES_DIM),
                                     'w' : np.zeros(USER_FEATURES_DIM),
                                     'updated' : False,
                                     'untouched_m' : True,
                                     'untouched_b' : True}

# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    if reward >= 0:
        current_article = ARTICLE_FEATURES[LAST_RECOMMENDATION]
        current_article['m'] += np.outer(LAST_USER, LAST_USER)
        current_article['invm'] = linalg.inv(current_article['m'],
                                             check_finite=False)
        if reward == 1:
            current_article['b'] += LAST_USER
            current_article['utouched_b'] = False
        current_article['updated'] = True
        current_article['untouched_m'] = False

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global LAST_RECOMMENDATION, LAST_USER, LAST_USER_NORM
    LAST_USER = np.array(user_features, copy=False)
    LAST_USER_NORM = np.linalg.norm(LAST_USER)
    LAST_RECOMMENDATION = max(articles, key=calculate_ucb)
    return LAST_RECOMMENDATION

def calculate_ucb(art_id):
    current_article = ARTICLE_FEATURES[art_id]
    if current_article['untouched_m']:
        ucb = ALPHA*LAST_USER_NORM
        return ucb
    if not current_article['untouched_b'] and current_article['updated']:
        current_article['w'] = current_article['invm'].dot(current_article['b'])
        current_article['updated'] = False
    if current_article['untouched_b']:
        ucb = ALPHA*math.sqrt(np.inner(LAST_USER,
                                       current_article['invm'].dot(LAST_USER)))
    else:
        ucb = np.inner(current_article['w'], LAST_USER) \
            + ALPHA*math.sqrt(np.inner(LAST_USER,
                                       current_article['invm'].dot(LAST_USER)))
    return ucb
