#!/usr/bin/env python3

import argparse

from tensorflow import keras as ks

import utils


def train(embed_size: int, batch_size: int, epochs: int) -> None:
    dataset = utils.Dataset()

    model = utils.build_model(
        sequence_len=dataset.sequence_len,
        vocab_size=dataset.vocab_size,
        embed_size=embed_size,
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        dataset.x_train,
        dataset.y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=[dataset.x_test, dataset.y_test],
        callbacks=[
            ks.callbacks.ModelCheckpoint(
                filepath=utils.get_filename(embed_size),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
            ),
        ],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training a model')
    parser.add_argument('-e', '--embedding-size', type=int, default=128,
                        help='Number of dimensions in the word embeddings')
    parser.add_argument('-n', '--num-epochs', type=int, default=10,
                        help='Number of epochs for which to fit the model')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Batch size for training the network')
    args = parser.parse_args()

    train(
        embed_size=args.embedding_size,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
    )
