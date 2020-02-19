import csv
import configuration as config


def get_data(filename, outList, outMap, mapTo):
    with open(filename) as f:
        output = f.readlines()
        output = list(set([x.strip() for x in output]))
        for i in output:
            if len(i) > 2:
                outList.append(i)
                outMap[i] = mapTo

def get_custom_data(filename=config.custom_test_set):
    word_map = {}
    with open(filename) as file:
        reader = csv.reader(file, delimiter= ',')
        for row in reader:
            word_map[row[0]] = 1 if row[1].lower() == 'e' else 0
    return word_map