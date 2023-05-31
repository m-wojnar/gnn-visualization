# gnn-visualization

## Installation

In order to install the package and its dependencies, run the following commands:

```bash
# clone the repository
git clone git@github.com:m-wojnar/gnn-visualization.git
cd gnn-visualization

# create a virtual environment and install the dependencies
python3 -m venv venv
pip install -r requirements.txt
```

**Note:** you have to install `faiss` package manually.

## Usage

### Faiss index creation

By default, the `FaissGenerator` class uses the `OpenML` platform to download the datasets. You can specify the dataset
you want to use by passing its id to the `FaissGenerator` constructor. The `FaissGenerator` class also takes the
`nn` and `rn` parameters, which specifies the number nearest and random neighbors. The `metric` parameter specifies 
the metric used to compute the nearest neighbors. `examples` sets the number of nodes to take from the graph for 
each of `n_graphs` graphs.

```python
from gnn_visualization.data import FaissGenerator

generator = FaissGenerator(dataset_id=554, nn=2, rn=1, metric='binary', examples=100, n_graphs=10)
generator.run()
```

### Saving and loading the results

You can save the results with pickle and lz4 compression

```python
generator.save('mnist_784/10g_100ex_binary_2nn_1rn.pkl.lz4')
```

and load them later to:

- a list of numpy arrays

    ```python
    graphs = FaissGenerator.load('mnist_784/10g_100ex_binary_2nn_1rn.pkl.lz4')
    ```

- a list of PyTorch tensors

    ```python
    graphs = FaissGenerator.load_torch('mnist_784/10g_100ex_binary_2nn_1rn.pkl.lz4', device)
    ```
  
- a PyToch geometric dataset

    ```python
    dataset = FaissGenerator.load_dataset('mnist_784/binary_full_nn2_rn1.pkl.lz4', device, batch_size=16, shuffle=True)
    ```