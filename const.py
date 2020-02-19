import torch
import numpy as np
import configuration as config
word_map = {}
english_words = []
french_words = []

last_letters = []
di_suffix = []
tri_suffix = []

bigrams = []
trigrams = []

training_set = {}
cv_set = {}
test_set = {}

featureCount = 0

X_train = np.zeros(shape=(featureCount, config.training_set_size)).transpose
y_train = np.zeros(config.training_set_size)

X_cv = np.zeros(shape=(featureCount, config.cv_set_size))
y_cv = np.zeros(config.cv_set_size)

X_test = np.zeros(shape=(featureCount, config.test_set_size))
y_test = np.zeros(config.test_set_size)

featureList = []




