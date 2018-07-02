#!/usr/bin/env python3

import argparse
import json
import os

from annoy import AnnoyIndex
from matplotlib import pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

import utils
from utils import Matrix


def get_index(embeddings: Matrix, num_trees: int) -> AnnoyIndex:
    embedding_size = np.shape(embeddings)[1]
    fname = 'embeddings_{}_dim_{}_trees.ann'.format(
        embedding_size, num_trees)
    index = AnnoyIndex(embedding_size)
    if os.path.exists(fname):
        index.load(fname)
        return index
    for i, vec in enumerate(embeddings):
        index.add_item(i, vec)
    index.build(num_trees)
    index.save(fname)
    return index


def dimensionality_reduction(embeddings: Matrix, convs: Matrix) -> None:
    reduction = PCA(n_components=2)
    reduction.fit(embeddings)
    tsne_embs = reduction.transform(embeddings)
    conv_embs = reduction.transform(convs.reshape(-1, convs.shape[-1]))
    plt.plot(tsne_embs[:, 0], tsne_embs[:, 1], '.')
    plt.plot(conv_embs[:, 0], conv_embs[:, 1], '.')
    plt.show()


def conv_knn(embeddings: Matrix,
             convs: Matrix,
             dense: Matrix,
             dataset: utils.Dataset,
             num_trees: int,
             num_nns: int,
             output_fname: str) -> None:
    index = get_index(embeddings, num_trees)

    def parse_vec(seq_id, filter_id):
        v = convs[seq_id, filter_id, :]
        idxs, dists = index.get_nns_by_vector(
            v, num_nns, include_distances=True)
        words = dataset.decode(idxs)
        return {'words': words, 'distances': dists}

    json.dump([
        {
            'weight': float(dense[j, 0]),
            'neighbors': [parse_vec(i, j) for i in range(convs.shape[0])],
        } for j in range(convs.shape[1])
    ], open(output_fname, 'w'), indent=2)


def visualize(embed_size: int,
              num_nns: int,
              num_trees: int,
              output_fname: str) -> None:
    embed_size = embed_size
    num_trees = num_trees

    fname = utils.get_filename(embed_size)
    if not os.path.exists(fname):
        raise RuntimeError('No such trained model exists: "{}". Run the '
                           'training script again, specifying embedding '
                           'size of {}.'.format(fname, embed_size))

    dataset = utils.Dataset()

    model = utils.build_model(
        sequence_len=dataset.sequence_len,
        vocab_size=dataset.vocab_size,
        embed_size=embed_size,
    )
    model.load_weights(fname)

    embeddings = model.get_layer('embeddings').get_weights()[0]
    convs = model.get_layer('convs').get_weights()[0].transpose(0, 2, 1)
    dense_weights = model.get_layer('output').get_weights()
    dense = dense_weights[0] - dense_weights[1]

    # Computes convolutional nearest neighbors.
    conv_knn(embeddings, convs, dense, dataset,
             num_trees, num_nns, output_fname)

    # Plots dimensionality reduction.
    dimensionality_reduction(embeddings, convs)

    plt.hist(embeddings.reshape(-1))
    plt.hist(convs.reshape(-1))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize trained network')
    parser.add_argument('-t', '--num-trees', type=int, default=10,
                        help='Number of trees to use in KNN model')
    parser.add_argument('-n', '--num-neighbors', type=int, default=10,
                        help='Number of neighbors to return')
    parser.add_argument('-e', '--embedding-size', type=int, default=128,
                        help='Number of dimensions in word embeddings')
    parser.add_argument('-s', '--save-loc', type=str, default='output.json',
                        help='Where to save the visualization JSON file')
    args = parser.parse_args()

    visualize(
        embed_size=args.embedding_size,
        num_nns=args.num_neighbors,
        num_trees=args.num_trees,
        output_fname=args.save_loc,
    )
