#!/usr/bin/env python3

import argparse
import os

import tensorflowjs as tfjs
from tensorflow import keras as ks

import utils
from utils import Dataset


def save_model_tfjs(model: ks.models.Model, save_loc: str) -> None:
    word_pred_model = ks.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('word_preds').output],
    )
    word_pred_model.compile(optimizer='sgd', loss='mse')
    tfjs.converters.save_keras_model(
        word_pred_model,
        save_loc,
    )


def save_accessory_json(dataset: Dataset, save_loc: str) -> None:
    utils.save_json(dataset.word_to_index, save_loc, 'word_to_index.json')


def productionize(save_loc: str, model_save_loc: str) -> None:
    if not os.path.exists(model_save_loc):
        raise RuntimeError('No such trained model exists: "{}". Run the '
                           'training script first!'.format(model_save_loc))

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    dataset = Dataset()

    model = ks.models.load_model(model_save_loc)

    print('Converting model to Tensorflow-JS format')
    save_model_tfjs(model, save_loc)

    print('Saving accessory JSON files')
    save_accessory_json(dataset, save_loc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to convert things to web-viewable format')
    parser.add_argument('-o', '--save-loc', type=str, default='web/',
                        help='Where to save the output files')
    parser.add_argument('-m', '--model-save-loc', type=str, default='model.h5',
                        help='Where the trained model is saved')
    args = parser.parse_args()

    productionize(
        save_loc=args.save_loc,
        model_save_loc=args.model_save_loc,
    )
