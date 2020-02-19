import re
import unicodedata

#script for parsing the most common french words
def strip_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

with open("text_samples/frenchdata.txt") as f1:
    lines = f1.readlines()
    lines = [strip_accents(line.strip().split()[0]).lower() for line in lines]
    print(lines)
    with open("text_samples/french.txt", 'w') as f2:
        f2.writelines(["%s\n" % line for line in lines])

