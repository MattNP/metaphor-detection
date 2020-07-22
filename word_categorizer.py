from empath import Empath

def categorize_words(sentences):
    lexicon = Empath()
    for s in sentences:
        lexicon.analyze(s, normalize=True)