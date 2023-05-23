from argparse import ArgumentParser

import torch
from sklearn.manifold import MDS

from data.faiss_generator import FaissGenerator
from models.train import create_graph, VisGNN
from utils.local_score import LocalMetric
from utils import ROOT_PATH


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

    mds = MDS(n_components=2, dissimilarity='euclidean')
    mds_vis = mds.fit_transform(X)

    metrics.calculate_knn_gain_and_dr_quality(mds_vis, X, y, 'MDS', dataset_size=0.95)
    metrics.visualize()
