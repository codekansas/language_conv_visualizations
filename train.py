#!/usr/bin/env python3

import argparse

from tensorflow import keras as ks

import utils


def build_model(sequence_len: int,
                vocab_size: int,
                embed_size: int,
                num_convolutions: int,
                conv_length: int) -> ks.models.Model:
    i = ks.layers.Input(shape=(sequence_len,))

    # The initializer is set so that the relative magnitude of the embeddings
    # are close to the relative magnitude of the convolutioanl filters, which
    # greatly improves training speed.
    x = ks.layers.Embedding(
        vocab_size,
        embed_size,
        embeddings_initializer=ks.initializers.RandomNormal(stddev=0.05),
        name='embeddings',
    )(i)

    # The convolutional layer is regularized to prevent all of the Convolutions
    # fitting to the same pattern.
    x = ks.layers.Conv1D(
        num_convolutions, 3,
        kernel_initializer=ks.initializers.RandomNormal(stddev=0.05),
        activity_regularizer=ks.regularizers.l2(0.001),
        use_bias=False,
        # activation='relu',
        name='convs',
    )(x)
    x = ks.layers.BatchNormalization()(x)

    x = ks.layers.Conv1D(
        1, 1,
        use_bias=False,
        activation='sigmoid',
        name='word_preds',
    )(x)
    x = ks.layers.GlobalAveragePooling1D()(x)
    return ks.models.Model(inputs=[i], outputs=[x])


def train(embed_size: int,
          batch_size: int,
          epochs: int,
          num_convolutions: int,
          conv_length: int,
          model_save_loc: str) -> None:
    dataset = utils.Dataset()

    model = build_model(
        sequence_len=dataset.sequence_len,
        vocab_size=dataset.vocab_size,
        embed_size=embed_size,
        num_convolutions=num_convolutions,
        conv_length=conv_length,
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.save(model_save_loc)

    model.fit(
        dataset.x_train,
        dataset.y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=[dataset.x_test, dataset.y_test],
        callbacks=[
            ks.callbacks.ModelCheckpoint(
                filepath=model_save_loc,
                monitor='val_loss',
                save_best_only=True,
            ),
        ],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training a model')
    parser.add_argument('-e', '--embedding-size', type=int, default=32,
                        help='Number of dimensions in the word embeddings')
    parser.add_argument('-n', '--num-epochs', type=int, default=10,
                        help='Number of epochs for which to fit the model')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Batch size for training the network')
    parser.add_argument('-c', '--num-convolutions', type=int, default=32,
                        help='Number of convolutional filters')
    parser.add_argument('-l', '--conv-length', type=int, default=3,
                        help='The length of the convolutional filters')
    parser.add_argument('-m', '--model-save-loc', type=str, default='model.h5',
                        help='Where to save the model during training')
    args = parser.parse_args()

    train(
        embed_size=args.embedding_size,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        num_convolutions=args.num_convolutions,
        conv_length=args.conv_length,
        model_save_loc=args.model_save_loc,
    )
