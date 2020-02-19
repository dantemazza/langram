import const
import dataParser
from functions import *
from features.grams import *
import configuration as config
import numpy as np
from features.extractFeatures import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import const



# extract the data from csv
# const.M_names, const.F_names, const.name_map = dataParser.get_data()
dataParser.get_data("text_samples/english.txt", const.english_words, const.word_map, 0)
dataParser.get_data("text_samples/french.txt", const.french_words, const.word_map, 1)
duplicates = [x for x in const.english_words if x in const.french_words]
const.english_words = [x for x in const.english_words if x not in duplicates]
const.french_words = [x for x in const.french_words if x not in duplicates]
for i in duplicates:
    if const.word_map.get(i):
        del const.word_map[i]
#extract the training/test sets
const.ngram_training_set = extract(config.data_extraction_size)
const.training_set = extract(config.training_set_size, labelled=True)
const.cv_set = extract(config.cv_set_size, labelled=True)
const.test_set = extract(config.test_set_size, labelled=True)
#determine most common ngrams

getCommonGrams(const.ngram_training_set)
get_suffixes(const.ngram_training_set)

const.featureCount += config.di_num + config.tri_num + config.last_letters + config.di_sufnum + config.tri_sufnum

const.X_train, const.y_train = extractFeatures(const.training_set)
const.X_cv, const.y_cv = extractFeatures(const.cv_set)
const.X_test, const.y_test = extractFeatures(const.test_set)


#--CLASSIFER--#

X_train = torch.stack([torch.tensor(i) for i in const.X_train])
y_train = torch.from_numpy(const.y_train)

X_cv = torch.stack([torch.tensor(i) for i in const.X_cv])
y_cv = torch.from_numpy(const.y_cv)

X_test = torch.stack([torch.tensor(i) for i in const.X_test])
y_test = torch.from_numpy(const.y_test)

training_set = data.TensorDataset(X_train, y_train)
training_loader = data.DataLoader(training_set, batch_size=config.minibatch, shuffle=True)

cv_set = data.TensorDataset(X_cv, y_cv)
cv_loader = data.DataLoader(cv_set, batch_size=config.minibatch, shuffle=False)

test_set = data.TensorDataset(X_test, y_test)
test_loader = data.DataLoader(test_set, batch_size=config.minibatch, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(const.featureCount, 1)

    def forward(self, X):
        return self.linear(X)

model = Model()
cost = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
epochs = int(config.iterations / config.training_set_size * config.minibatch)

iterations = 0
for i in range(epochs):
    for words, languages in training_loader:
        words = words.view(-1, const.featureCount).requires_grad_()
        optimizer.zero_grad()
        hypothesis = torch.sigmoid(model(words.float()))
        loss = cost(hypothesis.reshape(config.minibatch), languages.float())
        loss.backward()
        optimizer.step()
        iterations += 1
        if not iterations % config.print_interval:
           for num, set in enumerate([cv_loader, test_loader]):
                correct = 0
                total = 0
                for word, language in set:
                    word = word.view(-1, const.featureCount).requires_grad_()
                    pred = torch.sigmoid(model(word.float()))

                    total += language.size(0)
                    correct += ((pred.reshape(config.minibatch)-language.reshape(config.minibatch)).abs_() < 0.5).sum()

                accuracy = 100 * correct.item() / total
                type = "Test" if num else "CV"
                print('Type: {}- Iteration: {}. Cost: {}. Accuracy: {}'.format(type, iterations, loss.item(), accuracy))
            # print("")

if config.IS_DEBUG:
    for word, param in model.named_parameters():
        if param.requires_grad and config.IS_DEBUG:
            print(word)
            weights = param.data
            for feature, weight in zip(const.featureList, weights[0]):
                print(f"[{feature}] -> {weight}")
            config.IS_DEBUG = False

#now we can test custom data
if config.IS_CUSTOM:
    word_map_custom = dataParser.get_custom_data()

    X_custom, y_custom = extractFeatures(word_map_custom)

    X_custom_tensor = torch.stack([torch.tensor(i) for i in X_custom])
    y_custom_tensor = torch.from_numpy(y_custom)

    custom_set = data.TensorDataset(X_custom_tensor, y_custom_tensor)
    custom_loader = data.DataLoader(custom_set, batch_size=config.minibatch)


    for word, language in custom_loader:
        name = name.view(-1, const.featureCount).requires_grad_()
        pred = torch.sigmoid(model(name.float()))

        c_total = language.size(0)
        predictions = (pred.reshape(len(word_map_custom)) - language.reshape(len(word_map_custom))).abs_() < 0.5
        for index, name in enumerate(word_map_custom.keys()):
            print('Word: {}. Language: {}. Prediction: {}'.format(name, "FRE" if language[index].item() else "ENG", "FRE" if predictions[index].item() else "ENG"))
