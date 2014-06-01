#pylint: disable=C0301,C0111,W0603,W0613,E1101,E0611
#!/usr/bin/env python2.7

import math
import random
import numpy as np
from scipy import linalg


ARTICLE_FEATURES = {}
USER_FEATURES_DIM = 6
ARTICLE_FEATURES_DIM = 6
LAST_RECOMMENDATION = None
LAST_USER = None
ALPHA = 1.2

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    for art_key in art:
        ARTICLE_FEATURES[art_key] = {'features' : np.array(art[art_key],
                                                           copy=False),
                                     'm' : np.identity(USER_FEATURES_DIM*
                                                       ARTICLE_FEATURES_DIM),
                                     'b' : np.zeros(USER_FEATURES_DIM*
                                                    ARTICLE_FEATURES_DIM),
                                     'w' : np.zeros(USER_FEATURES_DIM*
                                                    ARTICLE_FEATURES_DIM),
                                     'untouched_m' : True,
                                     'untouched_b' : True}

# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    if reward >= 0:
        current_article = ARTICLE_FEATURES[LAST_RECOMMENDATION]
        mixed_model_vector = np.outer(LAST_USER,
                                      current_article['features']).ravel()
        current_article['m'] += np.outer(mixed_model_vector, mixed_model_vector)
        current_article['invm'] = linalg.inv(current_article['m'],
                                             check_finite=False)
        if reward == 1:
            current_article['b'] += mixed_model_vector
            current_article['utouched_b'] = False
        current_article['w'] = current_article['invm'].dot(current_article['b'])
        current_article['untouched_m'] = False

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global LAST_RECOMMENDATION, LAST_USER
    LAST_USER = np.array(user_features, copy=False)
    random.shuffle(articles)
    LAST_RECOMMENDATION = max(articles, key=calculate_ucb)
    return LAST_RECOMMENDATION

def calculate_ucb(art_id):
    current_article = ARTICLE_FEATURES[art_id]
    mixed_model_vector = np.outer(LAST_USER, current_article['features']).ravel()
    if current_article['untouched_m']:
        ucb = ALPHA*np.linalg.norm(mixed_model_vector)
        return ucb
    if current_article['untouched_b']:
        ucb = ALPHA*math.sqrt(np.inner(mixed_model_vector,
                                       current_article['invm']
                                       .dot(mixed_model_vector)))
    else:
        ucb = np.inner(current_article['w'], mixed_model_vector) \
            + ALPHA*math.sqrt(np.inner(mixed_model_vector,
                                       current_article['invm']
                                       .dot(mixed_model_vector)))
    return ucb
