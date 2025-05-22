# Function to get word indices given start and end character offsets
def get_word_indices(char_index_map, start, end):
    """
    Input:
        char_index_map: [(0, 4), (5, 13), (14, 17), (18, 23)]
        start: 0
        end: 4
    Output:
        result_indices: [0]
    """
    result_indices = []
    for i, (w_start, w_end) in enumerate(char_index_map):
        # Check if there is an overlap between [start, end) and [w_start, w_end)
        if not (end <= w_start or start >= w_end):
            result_indices.append(i)
    return result_indices

def get_char_index_map(sentence):
    """
    Precompute the start character indices of each word to speed lookup.
    Input:
        sentence: 'ross produced the album'
        
    Output:
        char_index_map: [(0, 4), (5, 13), (14, 17), (18, 23)]
    """
    # Split the full text into words
    words = sentence.split()
    
    char_index_map = []
    current_char_index = 0
    for w in words:
        char_index_map.append((current_char_index, current_char_index + len(w)))  # (start_char, end_char) for the word
        current_char_index += len(w) + 1  # +1 for the space after the word
        
    return char_index_map
        
def indexing_word(sentence, char_labels):
    """
    Process all annotations in sentence.
    Input:
        sentence: 'ross produced the album'
        
    Output:
        all_word_indices: [0]
    """
    char_index_map = get_char_index_map(sentence)

    all_word_indices = []
    for ann in char_labels:
        start_char = ann['start']
        end_char = ann['end']
        # Process one annotation in s
        word_indices = get_word_indices(char_index_map, start_char, end_char)
        all_word_indices+=(word_indices)

    return all_word_indices