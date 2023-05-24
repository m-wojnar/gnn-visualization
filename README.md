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
the metric used to compute the nearest neighbors and `examples` sets the number of examples to take from the dataset.

```python
from gnn_visualization.data import FaissGenerator

generator = FaissGenerator(dataset_id=554, nn=2, rn=1, metric='binary')
X, y, distances, indexes, n_neighbours = generator.run()
```

You can save the results with pickle and lz4 compression:

```python
generator.save('mnist_784/binary_full_nn2_rn1.pkl.lz4')
```

and load them later:

```python
X, y, distances, indexes, n_neighbours = FaissGenerator.load('mnist_784/binary_full_nn2_rn1.pkl.lz4')
```