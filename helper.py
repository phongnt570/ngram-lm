import re

def tokenize_english(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())

def tokenize_alphabetic(text):
    """Remove non-alphabetic characters. List all word tokens and normalize to lowercase."""
    text = ''.join([i for i in text if i.isalpha() or i.isspace()])
    return text.split()

def word_ngrams(text, n, tokenizer=None):
    """Given a sent as str return n-grams as a list of tuple"""
    
    # EXAMPLES 
    # > word_ngrams('hello world', 1)
    # [('hello',), ('world',)]
    # > word_ngrams('hello world', 2)
    # [('<s>', 'hello'), ('hello', 'world'), ('world', '</s>')]
    # > word_ngrams('hello world', 3)    
    # [('<s>', '<s>', 'hello'),  ('<s>', 'hello', 'world'), ('hello', 'world', '</s>')]
    # > word_ngrams('hello world', 4)
    # [('<s>', '<s>', '<s>', 'hello'), 
    # ('<s>', '<s>', 'hello', 'world'),
    # ('<s>', 'hello', 'world', '</s>')]
    
    if n <= 0:
        return None

    if tokenizer == None:
        tokenizer = tokenize_english
    tokens = tokenizer(text)

    if n == 1:
        return [(token,) for token in tokens]
    
    for _ in range(n-1):
        tokens.insert(0, '<s>')
    
    tokens.append('</s>')
    sequences = [tokens[i:] for i in range(n)]
    
    return [ngram for ngram in zip(*sequences)]
