from itertools import repeat, chain
from collections import Counter, defaultdict
from functools import partial
import re
import os
import json
import nltk
from scipy.sparse import dok_matrix
import numpy as np

NULL_WORD = "<null>"

class Corpus:
    def __init__(self, **kwargs):
        """
        `size`: How many words on each side to select for each window. Specifying
        `size` gives symmetric context windows and is equivalent to setting
        `left_size` and `right_size` to the same value.
        """
        if 'size' in kwargs:
            self.left_size = self.right_size = kwargs['size']
        elif 'left_size' in kwargs or 'right_size' in kwargs:
            self.left_size = kwargs.get('left_size', 0)
            self.right_size = kwargs.get('right_size', 0)
        else:
            raise KeyError("At least one of `size`, `left_size`, and `right_size` must be given")
        self._words = None
        self._word_index = None
        self._cooccurrence_matrix = None
    
    def tokenized_regions(self):
        """
        Returns an iterable of all tokenized regions of text from the corpus.
        """
        return map(self.tokenize, self.extract_regions())
            
    def fit(self):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(Counter)
        for region in self.tokenized_regions():
            word_counts.update(region)
            for left_context, word, right_context in self.region_context_windows(region):
                cooccurrence_counts[word].update(chain(left_context, right_context))
        self._words = [t[0] for t in word_counts.most_common()]
        self._word_index = {word: i for i, word in enumerate(self._words)}
        self._cooccurrence_matrix = dok_matrix((len(self._words), len(self._words)), dtype=np.uint32)
        self._cooccurrence_matrix.update(
            {(self._word_index[x], self._word_index[y]): count
              for x, counts in cooccurrence_counts.items() for y, count in counts.items()
              if y != NULL_WORD}
        )

    def is_fit(self):
        """
        Returns a boolean for whether or not the Corpus object has been fit to
        the text yet.
        """
        return self._words is not None

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
    
    @staticmethod
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

    def region_context_windows(self, region):
        for i, word in enumerate(region):
            start_index = i - self.left_size
            end_index = i + self.right_size
            left_context = self.window(region, start_index, i - 1)
            right_context = self.window(region, i + 1, end_index)
            yield (left_context, word, right_context)

    @property
    def context_windows(self):
        return (self.region_context_windows(region) for region in corpus.tokenized_regions())

    @property
    def words(self):
        if not self.is_fit():
            self.fit()
        return self._words
    
    @property
    def word_index(self):
        if not self.is_fit():
            self.fit()
        return self._word_index

    @property
    def cooccurrence_matrix(self):
        if self._cooccurrence_matrix is None:
            self.fit()
        return self._cooccurrence_matrix


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

