import argparse
import textract
import csv

# local imports
import tokenizer
import word_categorizer
import topic_generation
import svn_model

def train_model():
    sentences = []
    corpus_filepath = 'metaphor-corpus-1.0/metaphor-corpus.csv'
    with open(corpus_filepath) as csv_file:
        corpus_reader = csv.reader(csv_file)
        next(corpus_reader)  # ignore the header
        sentences = [row[0] for row in corpus_reader]
    word_categorizer.categorize_words(sentences)
    print(len(sentences))

def process_file(file_path):
    print('---')
    text = textract.process(file_path).decode('utf-8')
    print(f'Reading file: {file_path}')
    sentences = tokenizer.get_sentences(text)
    print(f'File has {len(sentences)} sentences')
    print('---')
    metaphors = svn_model.get_metaphors(sentences)
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
    parser = argparse.ArgumentParser(description='Process a file.')
    parser.add_argument('file', type=str, help='the path for the file to process')
    args = parser.parse_args()
    file_path = args.file
    process_file(file_path)