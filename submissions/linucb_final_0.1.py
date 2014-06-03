#pylint: disable=C0301,C0111,W0603,W0613,E1101,E0611
#!/usr/bin/env python2.7

import math
import random
import numpy as np
from scipy import linalg

ARTICLE_FEATURES = {}
USER_FEATURES_DIM = 6
LAST_RECOMMENDATION = None
LAST_USER = None
LAST_USER_NORM = None
ALPHA = 0.1

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    for art_key in art:
        ARTICLE_FEATURES[art_key] = {'features' : art[art_key],
                                     'm' : np.identity(USER_FEATURES_DIM),
                                     'b' : np.zeros(USER_FEATURES_DIM),
                                     'w' : np.zeros(USER_FEATURES_DIM),
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
            current_article['untouched_b'] = False
        current_article['w'] = current_article['invm'].dot(current_article['b'])
        current_article['untouched_m'] = False

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global LAST_RECOMMENDATION, LAST_USER, LAST_USER_NORM
    LAST_USER = np.array(user_features, copy=False)
    LAST_USER_NORM = np.linalg.norm(LAST_USER)
    #random.shuffle(articles)
    LAST_RECOMMENDATION = max(articles, key=calculate_ucb)
    return LAST_RECOMMENDATION

def calculate_ucb(art_id):
    current_article = ARTICLE_FEATURES[art_id]
    if current_article['untouched_m']:
        ucb = ALPHA*LAST_USER_NORM
        return ucb
    if current_article['untouched_b']:
        ucb = ALPHA*math.sqrt(LAST_USER.dot(current_article['invm']
                                            .dot(LAST_USER)))
    else:
        ucb = current_article['w'].T.dot(LAST_USER) \
            + ALPHA*math.sqrt(LAST_USER.dot(current_article['invm']
                                            .dot(LAST_USER)))
    return ucb
