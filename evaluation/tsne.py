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
    parser.add_argument('--n_iter', default = 7500, type = int, help = 'Number of iterations for tSNE')
    parser.add_argument('--input_folder', default='features', type=str, help='Input folder having the features arrays')
    parser.add_argument('--input_filename', type=str, default='', help='features and labels file name')
    parser.add_argument('--class_selection', type=str, default='ordinal', help='class selection mode, ordinal or random')
    args = parser.parse_args()

    # Load arrays
    print('Loading arrays...')
    features_file = args.input_filename + '_features.npy'
    labels_file = args.input_filename + '_labels.npy'

    X = np.load(os.path.join(args.input_folder, features_file))
    T = np.load(os.path.join(args.input_folder, labels_file))
    print('Loaded: {}'.format(features_file))
    print('Loaded: {}'.format(labels_file))

    # Choose some of the classes (e.g. 10, 15, 20, etc.)
    num_classes = np.unique(T).shape[0]

    # Choose random num tsne_classes
    if args.class_selection == 'random':
        random_classes = random.sample(range(int(T[0]), int(T[-1])), args.tsne_classes)
        random_classes = np.array(random_classes).astype(np.float32)

        X_few = []
        T_few = []

        for class_idx in random_classes:
            X_temp = X[T==class_idx]
            T_temp = T[T==class_idx]

            X_few.append(X_temp)
            T_few.append(T_temp)
        
        X_few = np.concatenate(X_few)
        T_few = np.concatenate(T_few)

    # Choose first num tsne_classes
    elif args.class_selection == 'ordinal':
        X_few = X[T<=num_classes+args.tsne_classes-1]
        T_few = T[T<=num_classes+args.tsne_classes-1]

    # tSNE and its parameters
    tsne = TSNE(n_components=args.n_components, perplexity=args.perplexity, n_iter=args.n_iter, init='pca', verbose=9)
    tsne_proj = tsne.fit_transform(X_few)

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

        if args.class_selection == 'random':
            indices = T_few == int(random_classes[lab])
        elif args.class_selection == 'ordinal':
            indices = T_few == lab+num_classes
        
        if args.class_selection == 'random':
            ax.scatter(tsne_proj[indices, 0],
                tsne_proj[indices, 1],
                c=np.array(cmap[i][0](color)).reshape(1, 4),
                #label=CUBClassNames.IDS_TO_NAMES[lab],
                label=int(random_classes[lab]),
                alpha=0.5)
        elif args.class_selection == 'ordinal':
            ax.scatter(tsne_proj[indices, 0],
                tsne_proj[indices, 1],
                c=np.array(cmap[i][0](color)).reshape(1, 4),
                #label=CUBClassNames.IDS_TO_NAMES[lab],
                label=lab+num_classes,
                alpha=0.5)

    ax.legend(bbox_to_anchor=(1.15, 1.0), loc='upper right', prop={'size': 10}, markerscale=2)
    plt.tight_layout()

    # Create dir to save tSNE plots
    if not os.path.exists('tSNE'):
        os.makedirs('tSNE')
    
    # Save figure
    plt.savefig('tSNE/{}_{}_classes_{}_perplexity.png'.format(args.input_filename, args.tsne_classes, args.perplexity))