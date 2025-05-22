import epitran
import panphon.distance

epi = epitran.Epitran("eng-Latn", cedict_file='cedict_1_0_ts_utf-8_mdbg.txt')
dst = panphon.distance.Distance()

def normalized_edit_distance(dist, before, after):
    return 1- (dist / max(len(before), len(after)))

def calculate_distance(before, after):
    before = epi.transliterate(before)
    after = epi.transliterate(after)

    dist = dst.levenshtein_distance(before, after)
    
    dist = dst.levenshtein_distance(before, after)
    norm_dist = normalized_edit_distance(dist, before, after)
    
    return norm_dist