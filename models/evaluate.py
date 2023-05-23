from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import MDS

from data.faiss_generator import FaissGenerator
from models.train import create_graph, VisGNN
from utils.local_score import LocalMetric
from utils import ROOT_PATH, Timer


def scatter_mds(xs, ys, path):
    embedding = pd.DataFrame(xs, columns=["1st Dimension", "2nd Dimension"])
    embedding["label"] = ys
    plt.figure(figsize=(6, 6))

    for y in np.unique(ys):
        df = embedding[embedding["label"] == y]
        plt.scatter(df["1st Dimension"], df["2nd Dimension"], label=y, marker=".")
    
    plt.title("MDS")
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--data', type=str, default=f'{ROOT_PATH}/data/mnist_784/dataset_nn100.pkl.lz4')
    args = args.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y, distances, indexes, nn_val = FaissGenerator.load(args.data)
    X, y, distances, indexes = tuple(map(lambda x: torch.from_numpy(x).to(device), (X, y, distances[:, 1:], indexes[:, 1:])))
    graph = create_graph(X, y, distances, indexes, nn_val)
    X, y = X.detach().cpu().numpy(), y.detach().cpu().numpy()

    metrics = LocalMetric()

    for i in range(20, 101, 20):
        model = VisGNN(input_dim=X.shape[1], hidden_dim=64, num_layers=5).to(device)
        model.load_state_dict(torch.load(f'{ROOT_PATH}/models/checkpoints/vis_gnn_model_{i}.pt'))
        gnn_vis = model(graph.x, graph.edge_index, graph.edge_attr).detach().cpu().numpy()

        metrics.calculate_knn_gain_and_dr_quality(gnn_vis, X, y, f'VisGNN {i}', dataset_size=0.95)

    with Timer('Calculating MDS projection...'):
        mds = MDS(n_components=2, dissimilarity='euclidean')
        mds_vis = mds.fit_transform(X)
    scatter_mds(mds_vis, y, f'{ROOT_PATH}/models/checkpoints/vis_mds.pdf')
    

    metrics.calculate_knn_gain_and_dr_quality(mds_vis, X, y, 'MDS', dataset_size=0.95)
    metrics.visualize()
