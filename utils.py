#!/usr/bin/env python3

from typing import List

from tensorflow import keras as ks


def build_model(sequence_len, vocab_size, embed_size):
    i = ks.layers.Input(shape=(sequence_len,))
    x = ks.layers.Embedding(vocab_size, embed_size, name='embeddings')(i)
    x = ks.layers.Conv1D(1000, 3, activation='relu')(x)
    x = ks.layers.GlobalAveragePooling1D()(x)
    x = ks.layers.Dense(1, activation='sigmoid')(x)
    return ks.models.Model(inputs=[i], outputs=[x])


class Dataset(object):
    def __init__(self, sequence_len=400):
        self.word_to_index = ks.datasets.imdb.get_word_index()
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.sequence_len = sequence_len

    def decode(self, x: List[int]) -> str:
        return ' '.join(self.index_to_word.get(i - 3, 'X') for i in x[1:])

    def encode(self, x: str) -> List[int]:
        return [1] + [
            self.word_to_index.get(w, -1) + 3
            for w in ks.preprocessing.text.text_to_word_sequence(x)
        ]

    @property
    def data(self):
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
    def x_train(self): return self.data[0][0]

    @property
    def y_train(self): return self.data[0][1]

    @property
    def x_test(self): return self.data[1][0]

    @property
    def y_test(self): return self.data[1][1]

    @property
    def vocab_size(self):
        if not hasattr(self, '_vocab_size'):
            self._vocab_size = max(
                self.x_train.max(), self.y_train.max(),
                self.x_test.max(), self.y_test.max(),
            )
        return self._vocab_size
