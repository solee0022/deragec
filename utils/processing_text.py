import re
import random, copy
from utils.normalizers import EnglishTextNormalizer
from num2words import num2words

normalizer = EnglishTextNormalizer()

def processing(text):
    # text = text.replace("'s", "apostrophe s") # To distinguish between possessive ('s) and verb ('s)
    text = normalizer(text)
    text = re.sub(r"[-+]?\d*\.?\d+|\d+%?", lambda m: num2words(m.group()), text).replace('%', ' percent')
    # text = text.replace("apostrophe s", "'s")
    return text.strip()

def processing_nbest(nbest):
    # processing nbest
    input = []
    for result in nbest[0]:
        if len(input) < 5 and len(result) > 0:
            input.append(result)
    if len(input) < 5:
        for _ in range(5 - len(input)):
            repeat = copy.deepcopy(random.choice(input))
            input.append(repeat)

    for i in range(len(input)):
        # text = processing(input[i])
        text = input[i]
        input[i] = text if len(text) > 0 else '<UNK>'
    
    return input
