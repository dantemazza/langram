import const
import random as rd


def popall(name):
    const.word_map.pop(name)
    const.french_words.remove(name)
    const.english_words.remove(name)

#extract a subset of the dataset with an option for labelled
def extract(size, labelled=False):
    half = int(size/2)

    result = {} if labelled else []
    for i in range(half):
        m, f = len(const.french_words), len(const.english_words)
        index = rd.randint(0, f-1 if m > f else m-1)
        if labelled:
            if result.get(const.french_words[index]):
                print(const.french_words[index])
            elif result.get(const.english_words[index]):
                print(const.english_words[index])
            result[const.french_words[index]] = 1
            result[const.english_words[index]] = 0
        else:
            result.append(const.french_words[index])
            result.append(const.english_words[index])
        del const.french_words[index]
        del const.english_words[index]
    return result

