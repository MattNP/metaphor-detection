def get_sentences(text):
    sentences = text.replace('\n', '').split('.')
    return [s.strip() for s in sentences]
