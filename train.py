#!/usr/bin/env python3

import argparse

from tensorflow import keras as ks

import utils


def build_language_model(vocab_size: int,
                         embed_size: int,
                         num_convolutions: int,
                         conv_length: int) -> ks.models.Model:
    i = ks.layers.Input(shape=(None,))
    x = ks.layers.Embedding(
        vocab_size,
        embed_size,
        embeddings_initializer=ks.initializers.RandomNormal(stddev=0.05),
        name='embeddings',
    )(i)
    x = ks.layers.Conv1D(
        num_convolutions, conv_length,
        kernel_initializer=ks.initializers.RandomNormal(stddev=0.05),
        use_bias=False,
        name='convs',
    )(x)
    x = ks.layers.Conv1D(
        vocab_size, 1,
        activation='softmax',
        name='word_preds',
    )(x)
    model = ks.models.Model(inputs=[i], outputs=[x])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_prediction_model(vocab_size: int,
                           embed_size: int,
                           num_convolutions: int,
                           conv_length: int) -> ks.models.Model:
    i = ks.layers.Input(shape=(None,))
    x = ks.layers.Embedding(
        vocab_size,
        embed_size,
        embeddings_initializer=ks.initializers.RandomNormal(stddev=0.05),
        name='embeddings',
    )(i)
    x = ks.layers.Conv1D(
        num_convolutions, conv_length,
        kernel_initializer=ks.initializers.RandomNormal(stddev=0.05),
        padding='same',
        use_bias=False,
        name='convs',
    )(x)
    x = ks.layers.Conv1D(
        1, 1,
        name='word_preds',
    )(x)
    x = ks.layers.GlobalAveragePooling1D()(x)
    x = ks.layers.Activation('sigmoid')(x)
    model = ks.models.Model(inputs=[i], outputs=[x])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train(embed_size: int,
          batch_size: int,
          epochs: int,
          num_convolutions: int,
          conv_length: int,
          model_save_loc: str,
          train_language_model: bool) -> None:

    if train_language_model:
        dataset = utils.LanguageModelDataset(conv_length=conv_length)
        model = build_language_model(
            vocab_size=dataset.vocab_size,
            embed_size=embed_size,
            num_convolutions=num_convolutions,
            conv_length=conv_length,
        )
    else:
        dataset = utils.PredictionDataset()
        model = build_prediction_model(
            vocab_size=dataset.vocab_size,
            embed_size=embed_size,
            num_convolutions=num_convolutions,
            conv_length=conv_length,
        )

    model.summary()
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
    parser.add_argument('-c', '--num-convolutions', type=int, default=128,
                        help='Number of convolutional filters')
    parser.add_argument('-l', '--conv-length', type=int, default=3,
                        help='The length of the convolutional filters')
    parser.add_argument('-m', '--model-save-loc', type=str, default='model.h5',
                        help='Where to save the model during training')
    parser.add_argument('--train-language-model', default=False,
                        action='store_true',
                        help='If set, trains a language model')
    args = parser.parse_args()

    train(
        embed_size=args.embedding_size,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        num_convolutions=args.num_convolutions,
        conv_length=args.conv_length,
        model_save_loc=args.model_save_loc,
        train_language_model=args.train_language_model,
    )
