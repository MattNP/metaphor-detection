from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from glob import glob

def get_topics(sentences):
    dictionary = Dictionary(sentences)
    corpus = [dictionary.doc2bow(text) for text in sentences]
    lda = LdaModel(corpus, num_topics=10)
    print(lda)