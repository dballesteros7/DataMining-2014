#!/usr/bin/env python2.7

import math
import numpy as np

#pylint: disable=C0301,C0111,W0603,W0613
ARTICLE_FEATURES = {}
USER_FEATURES_DIM = 6
LAST_RECOMMENDATION = None
LAST_USER = None
ALPHA = 3

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    for art_key in art:
        ARTICLE_FEATURES[art_key] = {'features' : art[art_key],
                                     'm' : np.identity(USER_FEATURES_DIM),
                                     'b' : np.zeros(USER_FEATURES_DIM),
                                     'w' : np.zeros(USER_FEATURES_DIM),
                                     'updated' : False}


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    if reward == -1:
        return
    ARTICLE_FEATURES[LAST_RECOMMENDATION]['m'] += np.outer(LAST_USER,
                                                           LAST_USER)
    ARTICLE_FEATURES['updated'] = True
    if reward == 1:
        ARTICLE_FEATURES[LAST_RECOMMENDATION]['b'] += LAST_USER

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global LAST_RECOMMENDATION, LAST_USER
    LAST_USER = np.array(user_features)
    LAST_RECOMMENDATION = max(articles, key=calculate_ucb)
    return LAST_RECOMMENDATION

def calculate_ucb(art_id):
    current_article = ARTICLE_FEATURES[art_id]
    if current_article['updated']:
        current_article['w'] = np.linalg.solve(current_article['m'],
                                               current_article['b'])
        current_article['updated'] = False
    ucb = np.inner(current_article['w'], LAST_USER)
    ucb += ALPHA*math.sqrt(np.inner(LAST_USER, np.linalg.solve(current_article['m'], LAST_USER)))
    return ucb
