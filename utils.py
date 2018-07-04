#!/usr/bin/env python3

from typing import List, Tuple

import numpy as np
from numpy import ndarray as Matrix  # For typing
import tensorflow as tf
from tensorflow import keras as ks

# Defines some types that are used in various places.
DataPair = Tuple[Tuple[Matrix, Matrix], Tuple[Matrix, Matrix]]


def get_filename(embed_size: int) -> str:
    return 'model_{}_embed.h5'.format(embed_size)


def build_model(sequence_len: int,
                vocab_size: int,
                embed_size: int) -> ks.models.Model:
    i = ks.layers.Input(shape=(sequence_len,))
    x = ks.layers.Embedding(
        vocab_size,
        embed_size,
        embeddings_initializer=ks.initializers.RandomNormal(stddev=0.05),
        name='embeddings',
    )(i)
    x = ks.layers.Conv1D(100, 3, name='convs')(x)
    x = ks.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=2, keepdims=True),
        name='word_preds',
    )(x)
    x = ks.layers.GlobalAveragePooling1D()(x)
    x = ks.layers.Activation('sigmoid')(x)
    return ks.models.Model(inputs=[i], outputs=[x])


class Dataset(object):
    def __init__(self, sequence_len: int=400) -> None:
        self.word_to_index = ks.datasets.imdb.get_word_index()
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.sequence_len = sequence_len

    def decode(self, x: List[int]) -> List[str]:
        return [self.index_to_word.get(i - 3, 'X') for i in x[1:]]

    def encode(self, x: str) -> List[int]:
        return [1] + [
            self.word_to_index.get(w, -1) + 3
            for w in ks.preprocessing.text.text_to_word_sequence(x)
        ]

    @property
    def data(self) -> DataPair:
        if not hasattr(self, '_x_train'):
            (x_train, y_train), (x_test, y_test) = ks.datasets.imdb.load_data()
            x_train = ks.preprocessing.sequence.pad_sequences(
                x_train, maxlen=self.sequence_len)
            x_test = ks.preprocessing.sequence.pad_sequences(
                x_test, maxlen=self.sequence_len)
            self._x_train, self._y_train = x_train, y_train
            self._x_test, self._y_test = x_test, y_test
        return (self._x_train, self._y_train), (self._x_test, self._y_test)

    @property
    def x_train(self) -> Matrix: return self.data[0][0]

    @property
    def y_train(self) -> Matrix: return self.data[0][1]

    @property
    def x_test(self) -> Matrix: return self.data[1][0]

    @property
    def y_test(self) -> Matrix: return self.data[1][1]

    @property
    def vocab_size(self) -> int:
        if not hasattr(self, '_vocab_size'):
            self._vocab_size = max(
                np.max(self.x_train), np.max(self.y_train),
                np.max(self.x_test), np.max(self.y_test),
            ) + 1
        return self._vocab_size
