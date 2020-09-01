from empath import Empath

def categorize_words(sentences):
    lexicon = Empath()
    counts_cats = []
    for s in sentences:
        counts_cats.append(lexicon.analyze(s))
    return counts_cats
