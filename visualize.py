#!/usr/bin/env python3

import argparse
import os
from typing import Tuple, List

from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tensorflow import keras as ks

import utils
from utils import Dataset, Matrix  # Project-specific types


def get_ordered_sentences(words: List[str],
                          scores: List[float],
                          conv_len: int) -> List[Tuple[str, float]]:
    sentence_score_pairs = [
        (
            ' '.join(words[max(i - conv_len // 2, 0):i + (conv_len + 1) // 2]),
            scores[i],
        )
        for i in range(len(scores))
    ]
    sentence_score_pairs.sort(key=lambda x: x[1])  # Sorts by scores.
    return sentence_score_pairs


def make_examples(save_loc: str) -> None:
    word_preds = utils.load_json(save_loc, 'word_predictions.json')
    conv_len = word_preds['conv_len']
    word_preds = word_preds['predictions']
    with open(os.path.join(save_loc, 'word_predictions_show.txt'), 'w') as f:
        for example in word_preds:
            ordered_sentences = get_ordered_sentences(
                example['words'],
                example['preds'],
                conv_len,
            )
            f.write('\n'.join('{}: {}'.format(*g)
                    for g in ordered_sentences))
            f.write('\n\n')

    excitations = utils.load_json(save_loc, 'excitations.json')
    conv_len = excitations['conv_len']
    excitations = excitations['activations']
    with open(os.path.join(save_loc, 'excitations_show.txt'), 'w') as f:
        for example in excitations:
            for i in range(3):
                ordered_sentences = get_ordered_sentences(
                    example['words'],
                    example['preds'][i],
                    conv_len,
                )
                f.write('Layer {}\n'.format(i))
                f.write('\n'.join('{}: {}'.format(*g)
                        for g in ordered_sentences))
                f.write('\n\n')


def get_word_predictions(model: ks.models.Model,
                         num_sentences: int,
                         dataset: Dataset,
                         save_loc: str) -> None:
    conv_len = model.get_layer('convs').get_weights()[0].shape[0]
    word_pred_model = ks.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('word_preds').output],
    )
    preds = word_pred_model.predict(dataset.x_train[:num_sentences])

    def get_preds(idxs: Matrix, preds: Matrix):
        num_padding = np.sum(idxs == 0)
        idxs, preds = idxs[num_padding:], preds[num_padding + 1:]
        words = dataset.decode(idxs)
        return {
            'words': words,
            'preds': [float(i) for i in preds],
        }

    utils.save_json({
        'conv_len': conv_len,
        'predictions': [
            get_preds(i, p)
            for i, p in zip(dataset.x_train[:num_sentences], preds)
        ],
    }, save_loc, 'word_predictions.json')


def get_excitations(model: ks.models.Model,
                    num_sentences: int,
                    dataset: Dataset,
                    save_loc: str) -> None:
    conv_len = model.get_layer('convs').get_weights()[0].shape[0]
    conv_output_model = ks.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('convs').output],
    )
    preds = conv_output_model.predict(dataset.x_train[:num_sentences])

    def get_preds(idxs: Matrix, preds: Matrix):
        num_padding = np.sum(idxs == 0)
        idxs, preds = idxs[num_padding:], preds[num_padding + 1:]
        words = dataset.decode(idxs)
        return {
            'words': words,
            'preds': [[float(j) for j in i] for i in preds.T],
        }

    utils.save_json({
        'conv_len': conv_len,
        'activations': [
            get_preds(i, p)
            for i, p in zip(dataset.x_train[:num_sentences], preds)
        ],
    }, save_loc, 'excitations.json')


def dimensionality_reduction(embeddings: Matrix,
                             convs: Matrix,
                             save_loc: str) -> None:
    convs = convs.transpose(0, 2, 1)
    reduction = PCA(n_components=2)
    reduction.fit(embeddings)
    embs = reduction.transform(embeddings)
    conv_embs = reduction.transform(convs.reshape(-1, convs.shape[-1]))
    plt.plot(embs[:, 0], embs[:, 1], '.', label='Embeddings')
    plt.plot(conv_embs[:, 0], conv_embs[:, 1], '.', label='Convolutions')
    plt.legend()
    plt.xlabel('First principle component')
    plt.ylabel('Second principle component')
    plt.savefig(os.path.join(save_loc, 'dimensionality_reduction.png'))
    plt.close()


def histograms(embeddings: Matrix, convs: Matrix, save_loc: str) -> None:
    i = 1
    plt.figure()
    for w, l in [(embeddings, 'Embeddings'), (convs, 'Convolutions')]:
        plt.subplot(2, 1, i)
        plt.hist(w.reshape(-1), label=l)
        plt.legend()
        plt.xlabel('Value of weight')
        plt.ylabel('Count')
        i += 1
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, 'histograms.png'))
    plt.close()


def conv_knn(embeddings: Matrix,
             convs: Matrix,
             dataset: Dataset,
             num_trees: int,
             num_nns: int,
             save_loc: str,
             cache_index: bool) -> None:
    index = np.dot(convs.transpose(0, 2, 1), embeddings.transpose(1, 0))
    sorted_idxs = np.argsort(index, axis=-1)

    def parse_vec(seq_id, filter_id):
        v = convs[seq_id, :, filter_id]
        idxs = sorted_idxs[seq_id, filter_id][-num_nns:]
        words = dataset.decode(idxs, keep_first=True)
        return {
            'words': words,
            'norm': float(np.sqrt(np.sum(v ** 2))),
        }

    utils.save_json([
        [parse_vec(i, j) for i in range(convs.shape[0])]
        for j in range(convs.shape[2])
    ], save_loc, 'knns.json')


def get_examples(dataset: Dataset, save_loc: str, num_examples: int) -> None:
    utils.save_json([
        ' '.join(dataset.decode(s)) for s in dataset.x_test[:num_examples]
    ], save_loc, 'examples.json')


def visualize(num_nns: int,
              num_trees: int,
              save_loc: str,
              cache_index: bool,
              num_sentences: int,
              model_save_loc: str) -> None:
    if not os.path.exists(model_save_loc):
        raise RuntimeError('No such trained model exists: "{}". Run the '
                           'training script first!'.format(model_save_loc))

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    dataset = Dataset()

    model = ks.models.load_model(model_save_loc)

    embeddings = model.get_layer('embeddings').get_weights()[0]
    convs = model.get_layer('convs').get_weights()[0]

    print('Computing KNNs to convolutions')
    conv_knn(embeddings, convs, dataset, num_trees,
             num_nns, save_loc, cache_index)

    print('Plotting dimensionality reduction on embeddings and convolutions')
    dimensionality_reduction(embeddings, convs, save_loc)

    print('Plotting histograms of embedding and convolution weights')
    histograms(embeddings, convs, save_loc)

    print('Computing word-wise predictions for sample sentences')
    get_word_predictions(model, num_sentences, dataset, save_loc)

    print('Computing convolutional excitation for sample sentences')
    get_excitations(model, num_sentences, dataset, save_loc)

    print('Converting JSON to text examples')
    make_examples(save_loc)

    print('Getting some examples')
    get_examples(dataset, save_loc, 10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize trained network')
    parser.add_argument('-t', '--num-trees', type=int, default=100,
                        help='Number of trees to use in KNN model')
    parser.add_argument('-n', '--num-neighbors', type=int, default=5,
                        help='Number of neighbors to return in KNN search')
    parser.add_argument('-o', '--save-loc', type=str, default='outputs/',
                        help='Where to save the visualization files')
    parser.add_argument('--no-cache-index', default=True,
                        action='store_false',
                        help='If set, caches the index lookup')
    parser.add_argument('-s', '--num-sentences', type=int, default=10,
                        help='Number of example sentences to visualize')
    parser.add_argument('-m', '--model-save-loc', type=str, default='model.h5',
                        help='Where the trained model is saved')
    args = parser.parse_args()

    visualize(
        num_nns=args.num_neighbors,
        num_trees=args.num_trees,
        save_loc=args.save_loc,
        cache_index=not args.no_cache_index,
        num_sentences=args.num_sentences,
        model_save_loc=args.model_save_loc,
    )
