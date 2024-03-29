from argparse import ArgumentParser, Namespace
from functools import reduce, partial
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, CGConv
from tqdm import tqdm

from data import FaissGenerator
from utils import ROOT_PATH
from utils.visualization import generate_plot


class VisGNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            edge_dim: int = 1,
            output_dim: int = 2,
            aggr: str = 'mean'
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [CGConv(channels=hidden_dim, dim=edge_dim, aggr=aggr) for _ in range(num_layers - 2)] +
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.layers[0](x, edge_index).relu()
        x = reduce(lambda x, layer: layer(x, edge_index, edge_attr).relu(), self.layers[1:-1], x)
        x = self.layers[-1](x, edge_index)
        return x


def mds_loss(out: Tensor, graphs: Data) -> float:
    d = F.pairwise_distance(out[graphs.edge_index[0]], out[graphs.edge_index[1]]).view(-1, 1)
    return F.mse_loss(d, graphs.edge_attr)


def ivhd_loss(out: Tensor, graphs: Data, c: float) -> float:
    d = F.pairwise_distance(out[graphs.edge_index[0]], out[graphs.edge_index[1]]).view(-1, 1)
    return torch.where(graphs.edge_attr == 0., d ** 2, c * (1 - d) ** 2).mean()


def train(
        model: nn.Module,
        dataset: DataLoader,
        graph: Data,
        args: Namespace,
        loss_fn: Callable,
        loss_params: Dict
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = partial(loss_fn, **loss_params)

    for epoch in range(1, args.epochs + 1):
        loss_val = 0
        steps = 0

        for graphs in tqdm(dataset, ascii=True, desc=f'Epoch {epoch}'):
            out = model(graphs.x, graphs.edge_index, graphs.edge_attr)
            loss = criterion(out, graphs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            steps += 1

        print(f'loss = {loss_val / steps}', flush=True)

        if epoch % 5 == 0:
            model.eval()
            torch.save(model.state_dict(), f'{args.output_path}/model_{epoch}.pt')
            generate_plot(model, graph, f'VisGNN {epoch}', f'{args.output_path}/vis_{epoch}.pdf', args.show_plots)
            model.train()

    return model


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--data', type=str, default=f'{ROOT_PATH}/data/mnist_784/100g_4000ex_binary_2nn_1rn.pkl.lz4')
    args.add_argument('--output_path', type=str, default=f'{ROOT_PATH}/models/checkpoints')
    args.add_argument('--show_plots', action='store_true')
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--hidden_dim', type=int, default=32)
    args.add_argument('--num_layers', type=int, default=5)
    args.add_argument('--lr', type=float, default=3e-4)
    args.add_argument('--loss', type=str, choices=['mds', 'ivhd'], default='ivhd')
    args = args.parse_args()

    loss = {
        'mds': (mds_loss, {}),
        'ivhd': (ivhd_loss, {'c': 0.1})
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph, dataset = FaissGenerator.load_dataset(args.data, device, args.batch_size)

    model = VisGNN(graph.x.shape[1], args.hidden_dim, args.num_layers).to(device)
    model = train(model, dataset, graph, args, *loss[args.loss])
    model.eval()

    torch.save(model.state_dict(), f'{args.output_path}/model_final.pt')
    generate_plot(model, graph, 'VisGNN', f'{args.output_path}/vis_final.pdf', args.show_plots)
