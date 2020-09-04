import argparse
import textract
import csv
from collections import defaultdict
import random

# local imports
import tokenizer
import word_categorizer
import topic_generation
import svm_model
import lda
import pickle
import gensim


def train_model():
    sentences = defaultdict(list)
    corpus_filepath = 'metaphor-corpus-1.0/metaphor-corpus.csv'
    with open(corpus_filepath) as csv_file:
        corpus_reader = csv.reader(csv_file)
        next(corpus_reader)  # ignore the header
        for row in corpus_reader:
            sentences[row[8]].append(row[0])
    #counts_cats = word_categorizer.categorize_words(sentences)
    #print(sentences.keys())
    #print(counts_cats[0])
    text_data = []
    for _,v in sentences.items():
        for s in v:
            tokens = lda.prepare_text_for_lda(s)
            text_data.append(tokens)
    
    # create dict and corpus
    dictionary = gensim.corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    
    # create lda model
    NUM_TOPICS = 100
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=100)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics()
    for topic in topics:
        print(topic)

    new_doc = 'Just this morning I discovered that PayPal had shut off my PayPal account, frozen my money in the account and has flagged my account for review.'
    new_doc = lda.prepare_text_for_lda(new_doc)
    new_doc_bow = dictionary.doc2bow(new_doc)
    print(new_doc_bow)
    print(ldamodel.get_document_topics(new_doc_bow))

def process_file(file_path):
    print('---')
    text = textract.process(file_path).decode('utf-8')
    print(f'Reading file: {file_path}')
    sentences = tokenizer.get_sentences(text)
    print(f'File has {len(sentences)} sentences')
    print('---')
    metaphors = svm_model.get_metaphors(sentences)
    print('Processing file...')
    print('---')
    print('Summary')
    n_words = sum([len(s.split()) for s in sentences])
    print(f'Words: {n_words}')
    print(f'Metaphors: {len(metaphors)}')
    print('---')
    if metaphors:
        print('List of metaphors:')
        for metaphor,sents in metaphors.items():
            print(f'Metaphor: {metaphor}')
            print('Sentences:')
            for sent in sents:
                print(f'\t{sent}')
            print('\n')
    print('---')

if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Process a file.')
    #parser.add_argument('file', type=str, help='the path for the file to process')
    #args = parser.parse_args()
    #file_path = args.file
    #process_file(file_path)
    train_model()