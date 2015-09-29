from itertools import repeat, chain
from collections import Counter
import re
import os
import json
import nltk

NULL_WORD = "<null>"

class Corpus:
    def __init__(self):
        self._words = None
        self._word_index = None

    def extract_regions(self):
        """
        Returns an iterable of all regions of text (strings) in the corpus that
        should each be considered one contiguous unit. Messages, comments,
        reviews, etc.
        """
        raise NotImplementedError()

    @staticmethod
    def tokenize(string):
        """
        Takes strings that come from `extract_regions` and returns the tokens
        from that string.
        """
        raise NotImplementedError()
    
    def tokenized_regions(self):
        """
        Returns an iterable of all tokenized regions of text from the corpus.
        """
        return map(self.tokenize, self.extract_regions())

    def _build_word_index(self):
        tokens = chain.from_iterable(self.tokenized_regions())
        word_counts = Counter(tokens)
        self._words = [t[0] for t in word_counts.most_common()]
        self._word_index = {word: i for i, word in enumerate(self._words)}

    @property
    def words(self):
        if self._words is None:
            self._build_word_index()
        return self._words
    
    @property
    def word_index(self):
        if self._word_index is None:
            self._build_word_index()
        return self._word_index

    def index_built(self):
        """
        Returns a boolean for whether or not the word index has been built yet.
        """
        return self._words is not None

    
class RedditCorpus(Corpus):
    def __init__(self, path):
        """
        If `path` is a file, it will be treated as the only file in the corpus.
        If it's a directory, every file not starting with a dot (".") will be
        considered to be in the corpus.
        """
        super().__init__()
        if os.path.isdir(path):
            file_names =  filter(lambda p: not p.startswith('.'), os.listdir(path))
            self.file_paths = [os.path.join(path, name) for name in file_names]
        else:
            self.file_paths = [path]

    def extract_regions(self):
        # This is not exactly a rock-solid way to get the body, but it's ~2x as
        # fast as json parsing each line
        body_snatcher = re.compile(r"\{.*?(?<!\\)\"body(?<!\\)\":(?<!\\)\"(.*?)(?<!\\)\".*}")
        for file_path in self.file_paths:
            with open(file_path) as file_:
                for line in file_:
                    match = body_snatcher.match(line)
                    if match:
                        body = match.group(1)
                        if not body == '[deleted]':
                            yield body

    @staticmethod
    def tokenize(string):
        return nltk.wordpunct_tokenize(string.lower())
                    

def window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    front_nulls = repeat(NULL_WORD, abs(min(start_index, 0)))
    back_nulls = repeat(NULL_WORD, max(end_index - last_index, 0))
    selected_tokens = region[max(start_index, 0) : min(end_index, last_index) + 1]
    return list(chain(front_nulls, selected_tokens, back_nulls))
    

def context_windows(corpus, **kwargs):
    """
    `size`: How many words on each side to select for each window. Specifying
    `size` gives symmetric context windows and is equivalent to setting
    `left_size` and `right_size` to the same value.
    """
    if 'size' in kwargs:
        left_size = right_size = kwargs['size']
    elif 'left_size' in kwargs or 'right_size' in kwargs:
        left_size = kwargs.get('left_size', 0)
        right_size = kwargs.get('right_size', 0)
    else:
        raise KeyError("At least one of `size`, `left_size`, and `right_size` must be given")
    for region in corpus.tokenized_regions():
        for i, word in enumerate(region):
            start_index = i - left_size
            end_index = i + right_size
            left_context = window(region, start_index, i - 1)
            right_context = window(region, i + 1, end_index)
            yield (left_context, word, right_context)

