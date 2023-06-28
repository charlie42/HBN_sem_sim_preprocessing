from collections import Counter
from itertools import chain

def print_common_words(items, n=100): # Print most common words in items
    # Join all items into one string
    print(items)
    words = list(chain.from_iterable([str(item).split() for item in items]))
    word_counts = Counter(words)
    print(word_counts.most_common(n))