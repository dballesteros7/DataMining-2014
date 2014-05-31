#!/usr/bin/env python2.7

import math
import numpy as np
import numpy.matlib as npmatlib
import numpy.random

#pylint: disable=C0301,C0111,W0603,W0613
ARTICLE_FEATURES = {}
USER_FEATURES_DIM = 6
LAST_RECOMMENDATION = None
LAST_USER = None
ALPHA = 2.5

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    for art_key in art:
        ARTICLE_FEATURES[art_key] = {'features' : art[art_key],
                                     'm' : npmatlib.identity(USER_FEATURES_DIM),
                                     'b' : npmatlib.zeros((USER_FEATURES_DIM, 1)),
                                     'w' : npmatlib.zeros((USER_FEATURES_DIM, 1)),
                                     'updated' : False}


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global LAST_RECOMMENDATION
    global LAST_USER
    if reward == 0:
        ARTICLE_FEATURES[LAST_RECOMMENDATION]['m'] += LAST_USER*LAST_USER.T
        ARTICLE_FEATURES['updated'] = True
    elif reward == 1:
        ARTICLE_FEATURES[LAST_RECOMMENDATION]['m'] += LAST_USER*LAST_USER.T
        ARTICLE_FEATURES[LAST_RECOMMENDATION]['b'] += LAST_USER
        ARTICLE_FEATURES['updated'] = True
    LAST_RECOMMENDATION = None
    LAST_USER = None

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global LAST_RECOMMENDATION
    global LAST_USER
    LAST_USER = np.matrix([user_features]).T
    LAST_RECOMMENDATION = max(articles, key=calculate_ucb(LAST_USER))
    return LAST_RECOMMENDATION

def calculate_ucb(user_features):
    def _calculate_ucb(art_id):
        current_article = ARTICLE_FEATURES[art_id]
        if current_article['updated']:
            current_article['w'] = current_article['m'].I*current_article['b']
            current_article['updated'] = False
        ucb = (current_article['w'].T*user_features)[0,0]
        ucb += ALPHA*math.sqrt((user_features.T*current_article['m'].I*user_features)[0,0])
        return ucb
    return _calculate_ucb
