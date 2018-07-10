#!/usr/bin/env python3

import json
import os
from typing import List, Tuple, Any

import numpy as np
from numpy import ndarray as Matrix  # For typing
from tensorflow import keras as ks

# Defines some types that are used in various places.
DataPair = Tuple[Tuple[Matrix, Matrix], Tuple[Matrix, Matrix]]


def save_json(obj: Any, save_loc: str, fname: str) -> None:
    json.dump(obj, open(os.path.join(save_loc, fname), 'w'), indent=2)


def load_json(save_loc: str, fname: str) -> Any:
    return json.load(open(os.path.join(save_loc, fname), 'r'))


class Dataset(object):
    def __init__(self, vocab_size: int=20000) -> None:
        self.word_to_index = ks.datasets.imdb.get_word_index()
        self.word_to_index = {
            k: v
            for k, v in self.word_to_index.items()
            if v < vocab_size - 3
        }
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.vocab_size = vocab_size

    def decode(self, x: List[int], keep_first: bool=False) -> List[str]:
        x = [i for i in x if i != 0]
        return [
            self.index_to_word.get(i - 3, 'X')
            for i in (x if keep_first else x[1:])
        ]

    def encode(self, x: str) -> List[int]:
        return [1] + [
            self.word_to_index.get(w, -1) + 3
            for w in ks.preprocessing.text.text_to_word_sequence(x)
        ]

    @property
    def data(self) -> DataPair:
        raise NotImplementedError('Data serialization is implemented in '
                                  'subclasses, not the raw Dataset class.')

    @property
    def x_train(self) -> Matrix: return self.data[0][0]

    @property
    def x_test(self) -> Matrix: return self.data[1][0]

    @property
    def y_train(self) -> Matrix: return self.data[0][1]

    @property
    def y_test(self) -> Matrix: return self.data[1][1]


class PredictionDataset(Dataset):
    def __init__(self,
                 *args,
                 sequence_len: int=400,
                 **kwargs) -> None:
        super(PredictionDataset, self).__init__(*args, **kwargs)
        self.sequence_len = sequence_len

    @property
    def data(self) -> DataPair:
        if not hasattr(self, '_data'):
            (x_train, y_train), (x_test, y_test) = ks.datasets.imdb.load_data(
                # maxlen=self.sequence_len,
                num_words=self.vocab_size,
            )
            x_train = ks.preprocessing.sequence.pad_sequences(
                x_train, maxlen=self.sequence_len)
            x_test = ks.preprocessing.sequence.pad_sequences(
                x_test, maxlen=self.sequence_len)
            self._data = (x_train, y_train), (x_test, y_test)
        return self._data


class LanguageModelDataset(Dataset):
    def __init__(self,
                 conv_length,
                 *args,
                 sequence_len: int=50,
                 **kwargs) -> None:
        super(LanguageModelDataset, self).__init__(*args, **kwargs)
        self.conv_length = conv_length
        self.sequence_len = sequence_len

    def indices_to_training_set(self, idxs: List[List[Matrix]]) -> Matrix:
        c, s = self.conv_length, self.sequence_len
        x, y = zip(*[
            (idx[i:i + s], idx[i + c:i + s + 1])
            for idx in idxs
            if len(idx) > self.sequence_len
            for i in range(0, len(idx) - s - 1, s)
        ])
        x, y = np.asarray(x), np.expand_dims(np.asarray(y), -1)
        return x, y

    @property
    def data(self) -> DataPair:
        if not hasattr(self, '_data'):
            (x_train, _), (x_test, _) = ks.datasets.imdb.load_data(
                # maxlen=self.sequence_len,
                num_words=self.vocab_size,
            )
            x_train, y_train = self.indices_to_training_set(x_train)
            x_test, y_test = self.indices_to_training_set(x_test)
            self._data = (x_train, y_train), (x_test, y_test)
        return self._data
