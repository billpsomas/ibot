import numpy as np 
import os
import matplotlib.pyplot as plt
import argparse
import random

from sklearn.manifold import TSNE
from matplotlib import cm
from sklearn import utils

import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tSNE visualization of features')
    parser.add_argument('--tsne_classes', default=20, type=int, help = 'Number of classes to be used for tSNE')
    parser.add_argument('--n_components', type=int, default=2, help = 'Number of components for tSNE')
    parser.add_argument('--perplexity', default = 10.0, type = float, help = 'tSNE perplexity parameter')
    parser.add_argument('--n_iter', default = 50000, type = int, help = 'Number of iterations for tSNE')
    parser.add_argument('--input_folder', default='features', type=str, help='Input folder having the features arrays')
    parser.add_argument('--input_filename', type=str, default='', help='features and labels file name')
    args = parser.parse_args()

    # Load arrays
    print('Loading arrays...')
    iboff_features_file = 'iboff_' + args.input_filename + '_features.npy'
    iboff_labels_file = 'iboff_' + args.input_filename + '_labels.npy'

    ibatt_features_file = 'ibatt_' + args.input_filename + '_features.npy'
    ibatt_labels_file = 'ibatt_' + args.input_filename + '_labels.npy'

    X_iboff = np.load(os.path.join(args.input_folder, iboff_features_file))
    T_iboff = np.load(os.path.join(args.input_folder, iboff_labels_file))
    print('Loaded: {}'.format(iboff_features_file))
    print('Loaded: {}'.format(iboff_labels_file))

    X_ibatt = np.load(os.path.join(args.input_folder, ibatt_features_file))
    T_ibatt = np.load(os.path.join(args.input_folder, ibatt_labels_file))
    print('Loaded: {}'.format(ibatt_features_file))
    print('Loaded: {}'.format(ibatt_labels_file))

    # Choose some of the classes (e.g. 10, 15, 20, etc.)
    assert np.unique(T_iboff).shape[0] == np.unique(T_ibatt).shape[0]
    num_classes = np.unique(T_iboff).shape[0]

    random_classes = random.sample(range(int(T_iboff[0]), int(T_iboff[-1])), args.tsne_classes)
    random_classes = np.array(random_classes).astype(np.float32)

    X_iboff_few = []
    T_iboff_few = []

    for class_idx in random_classes:
        X_temp = X_iboff[T_iboff==class_idx]
        T_temp = T_iboff[T_iboff==class_idx]

        X_iboff_few.append(X_temp)
        T_iboff_few.append(T_temp)
    
    X_iboff_few = np.concatenate(X_iboff_few)
    T_iboff_few = np.concatenate(T_iboff_few)

    X_ibatt_few = []
    T_ibatt_few = []

    for class_idx in random_classes:
        X_temp = X_ibatt[T_ibatt==class_idx]
        T_temp = T_ibatt[T_ibatt==class_idx]

        X_ibatt_few.append(X_temp)
        T_ibatt_few.append(T_temp)
    
    X_ibatt_few = np.concatenate(X_ibatt_few)
    T_ibatt_few = np.concatenate(T_ibatt_few)

    # tSNE and its parameters
    tsne = TSNE(n_components=args.n_components, perplexity=args.perplexity, n_iter=args.n_iter, init='pca', verbose=9)
    tsne_proj_iboff = tsne.fit_transform(X_iboff_few)
    tsne_proj_ibatt = tsne.fit_transform(X_ibatt_few)

    # Color maps for visualization
    cmap1 = cm.get_cmap('tab20')
    cmap2 = cm.get_cmap('Set3')
    cmap = np.vstack((cmap1, cmap2))

    # Plot results
    fig, ax = plt.subplots(figsize=(8,8))

    for lab in range(args.tsne_classes):
        if lab < 20:
            i = 0
            color = lab
        else:
            i = 1
            color = lab - 20
        indices = T_iboff_few == int(random_classes[lab])
        ax.scatter(tsne_proj_iboff[indices, 0],
            tsne_proj_iboff[indices, 1],
            c=np.array(cmap[i][0](color)).reshape(1, 4),
            label=int(random_classes[lab]),
            alpha=0.5)
    ax.legend(bbox_to_anchor=(1.15, 1.0), loc='upper right', prop={'size': 10}, markerscale=2)
    plt.tight_layout()

    # Create dir to save tSNE plots
    if not os.path.exists('tSNE'):
        os.makedirs('tSNE')
    
    # Save figure
    plt.savefig('tSNE/iboff_{}_{}_classes_{}_perplexity.png'.format(args.input_filename, args.tsne_classes, args.perplexity))

    # Plot results
    fig, ax = plt.subplots(figsize=(8,8))

    for lab in range(args.tsne_classes):
        if lab < 20:
            i = 0
            color = lab
        else:
            i = 1
            color = lab - 20
        indices = T_ibatt_few == int(random_classes[lab])
        ax.scatter(tsne_proj_ibatt[indices, 0],
            tsne_proj_ibatt[indices, 1],
            c=np.array(cmap[i][0](color)).reshape(1, 4),
            label=int(random_classes[lab]),
            alpha=0.5)
    ax.legend(bbox_to_anchor=(1.15, 1.0), loc='upper right', prop={'size': 10}, markerscale=2)
    plt.tight_layout()

    # Create dir to save tSNE plots
    if not os.path.exists('tSNE'):
        os.makedirs('tSNE')
    
    # Save figure
    plt.savefig('tSNE/ibatt_{}_{}_classes_{}_perplexity.png'.format(args.input_filename, args.tsne_classes, args.perplexity))