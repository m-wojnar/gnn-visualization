from argparse import ArgumentParser

import torch
from sklearn.manifold import MDS

from data.faiss_generator import FaissGenerator
from models.train import VisGNN
from utils import ROOT_PATH, Timer
from utils.local_score import LocalMetric
from utils.visualization import plot


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--data', type=str, default=f'{ROOT_PATH}/data/mnist_784/100g_4000ex_binary_2nn_1rn.pkl.lz4')
    args.add_argument('--path', type=str, default=f'{ROOT_PATH}/models/checkpoints')
    args.add_argument('--dataset_size', type=float, default=0.15)
    args.add_argument('--hidden_dim', type=int, default=64)
    args.add_argument('--num_layers', type=int, default=5)
    args = args.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph, _ = FaissGenerator.load_dataset(args.data, device)
    X, y = graph.x.detach().cpu().numpy(), graph.y.detach().cpu().numpy()

    metrics = LocalMetric()

    for i in range(20, 101, 20):
        model = VisGNN(input_dim=X.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        model.load_state_dict(torch.load(f'{args.path}/model_{i}.pt'))

        gnn_vis = model(graph.x, graph.edge_index, graph.edge_attr).detach().cpu().numpy()
        plot(gnn_vis, y, f'VisGNN {i}', f'{args.path}/eval_vis_{i}.pdf')

        metrics.calculate_knn_gain_and_dr_quality(gnn_vis, X, y, f'VisGNN {i}', dataset_size=args.dataset_size)

    with Timer('Calculating MDS projection...'):
        mds_vis = MDS(dissimilarity='euclidean', n_jobs=-1).fit_transform(X)
        plot(mds_vis, y, 'MDS', f'{args.path}/eval_vis_mds.pdf')

    metrics.calculate_knn_gain_and_dr_quality(mds_vis, X, y, 'MDS', dataset_size=args.dataset_size)
    metrics.visualize()
