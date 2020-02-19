import torch
import numpy as np
import const


def extractFeatures(set):
    X = np.zeros(shape=(len(set), const.featureCount))
    y = np.zeros(len(set))

    for i, name in enumerate(set):
        fn = 0
        for letter in const.last_letters:
            X[i][fn] = 10 if name[-1] == letter else 0
            fn += 1
            const.featureList.append(f"Last letter of: {letter}")
        for gram in const.bigrams + const.trigrams:
            X[i][fn] = 10 if gram in name else 0
            fn += 1
            const.featureList.append(f"Contains: {gram}")
        for difix in const.di_suffix:
            X[i][fn] = 10 if name[-2:] == difix else 0
            fn += 1
            const.featureList.append(f"Suffix of: {difix}")
        for trifix in const.tri_suffix:
            X[i][fn] = 10 if name[-3:] == trifix else 0
            fn += 1
            const.featureList.append(f"Suffix of: {trifix}")
        y[i] = set[name]

    return X, y






