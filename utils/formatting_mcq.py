def formatting_mcq(datapoint, q, f_dict):

    numbering = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

    ## he [Blank] need it. # fill-in-the-blank sentence vs. cloze sentence
    masked_sentence = datapoint["s-query"][q]
    
    ## A: rarely, B: really, C: rally
    hyp_ne = list(datapoint["hyps-NE"][q][0].keys())
    retr_ne = list(datapoint["retr-NE"][q][0].keys())
    ne = hyp_ne + retr_ne
        
    ## Phonetic Score
    scores = list(datapoint["retr-NE"][q][0].values())
    scores = list(datapoint["hyps-NE"][q][0].values()) + scores 

    set_ne = []
    set_scores = []
    for n_idx, n in enumerate(ne):
        if n not in set_ne:
            set_ne.append(n)
            set_scores.append(scores[n_idx])
            
    ## === MCQ+PS+Def === ##    
    options = ""          
    for num, (n, ps) in enumerate(zip(set_ne, set_scores)):
        
        ## add named_entity's definition
        if n in list(f_dict.keys()):
            defini = f_dict[n].strip()
        elif n.lower() in list(f_dict.keys()):
            defini = f_dict[n.lower()].strip()
        elif n.capitalize() in list(f_dict.keys()):
            defini = f_dict[n.capitalize()].strip()
        else:
            defini = n
        
        defini_tmp = defini.split()[:20]
        defini_tmp = " ".join(defini_tmp).replace("\n", " ").strip()

        options += f"{numbering[num]}: {n} ({ps:.2f} | {defini_tmp})\n" 
    ## =============== ##     
        
    return masked_sentence, options