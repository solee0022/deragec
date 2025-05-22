import numpy as np
from utils.indexing import get_word_indices, get_char_index_map

pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours',
 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
 'yourselves', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom',
 'whose', 'which', 'what', 'anyone', 'anybody', 'someone', 'somebody',
 'everyone', 'everybody', 'no one', 'nobody', 'anything', 'something',
 'everything', 'nothing', 'each other', 'one another']

def masking_ne(tagger, sample):
    """_summary_
    Inputs:
        best_hyp
        tagger: NE tagger
    Returns:
        one_maskeds[w_idx]: a NE is masked in a sentence
        all_masked: all NEs are masked in a sentence
        ne: role as similar-sounding
        word_labels_ASR: NE word indices
    """
    contents = [] # return contents

    ERROR = {}
    s = sample["best_hyp"]
    # Split the full text into words
    words = s.split(" ")
    one_maskeds = []
    
    labels = ["person", "location", "organization"]
    char_labels_ASR = tagger.predict_entities(s, labels)
    char_index_map = get_char_index_map(s)

    # Process one annotation in s
    for ann in char_labels_ASR:
        word_indices = get_word_indices(char_index_map, ann["start"], ann["end"])

        # ne = " ".join(words[word_indices[0]:word_indices[-1]+1])
        ne = sample["best_hyp"][ann["start"]:ann["end"]].strip()
        
        if ne.lower() not in pronouns: # except for pronouns
            ERROR[word_indices[0]] = ne 

            # Masking ne in s
            one_masked = sample["best_hyp"][:ann["start"]] + "[BLANK]" + sample["best_hyp"][ann["end"]:]
            one_maskeds.append(one_masked)
    
    # Process all annotations in s   
    all_masked = words.copy() 
    for k, v in ERROR.items(): 
        word_cnt = len(v.split(" "))
        all_masked[k] = "[BLANK]"
        if word_cnt != 1:
            for i in range(1, word_cnt):
                all_masked[k+i] = "[BLANK*]"

    # all_masked = [m for m in all_masked if m != "[BLANK*]"]
    all_masked = list(filter(lambda a: a != "[BLANK*]", all_masked))
    all_masked = " ".join(all_masked)
    
    for w_idx, (k, ne) in enumerate(ERROR.items()):
        audio = sample['audio']
        custom_id = f'request///{audio}///{one_maskeds[w_idx]}///{all_masked}'
        contents.append([custom_id, one_maskeds[w_idx], ne, k])
                
    return contents        