#!/usr/bin/env python3

from typing import List

import numpy as np
from tensorflow import keras as ks


class Dataset(object):
    def __init__(self):
        self.train, self.test = ks.datasets.imdb.load_data()
        self.word_to_index = ks.datasets.imdb.get_word_index()
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

    def decode(self, x: List[int]) -> str:
        return ' '.join(self.index_to_word.get(i - 3, 'X') for i in x[1:])

    def encode(self, x: str) -> List[int]:
        return [1] + [
            self.word_to_index.get(w, -1) + 3
            for w in ks.preprocessing.text.text_to_word_sequence(x)
        ]

    @property
    def x_train(self) -> np.array: return self.train[0]

    @property
    def y_train(self) -> np.array: return self.train[1]

    @property
    def x_test(self) -> np.array: return self.test[0]

    @property
    def y_test(self) -> np.array: return self.test[1]
