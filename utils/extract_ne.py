from gliner import GLiNER


def extract_ne(tagger, labels, pronouns, text):
    char_labels_GT = tagger.predict_entities(text, labels)
    
    for k, en in enumerate(char_labels_GT):       
        ne = text[en["start"]:en["end"]]
        if ne.lower() not in pronouns:
            print(ne)      
            
# load NER model
tagger = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
labels = ["person", "location", "organization"] 
pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours',
 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
 'yourselves', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom',
 'whose', 'which', 'what', 'anyone', 'anybody', 'someone', 'somebody',
 'everyone', 'everybody', 'no one', 'nobody', 'anything', 'something',
 'everything', 'nothing', 'each other', 'one another']

# inference   
text = "sommerville shared the patent for the case extracting system with galand"
extract_ne(tagger, labels, pronouns, text)