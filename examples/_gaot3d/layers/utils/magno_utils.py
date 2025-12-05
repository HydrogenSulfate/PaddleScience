import importlib
from typing import Literal

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle_cluster import radius as paddle_cluster_radius


class NeighborSearch(nn.Layer):
    """
    Neighborhood search between two arbitrary coordinate meshes.
    For each point `x` in `queries`, returns a set of the indices of all points `y` in `data`
    within the ball of radius r `B_r(x)`

    Parameters
    ----------
    use_open3d : bool
        Whether to use open3d or native Pypaddle implementation
        NOTE: open3d implementation requires 3d data
    use_paddle_cluster: bool
        Whether to use paddle_cluster for finding neighbors.
    """

    def __init__(self, use_paddle_cluster=False):
        super().__init__()
        if use_paddle_cluster:
            self.search_fn = self._paddle_cluster_search
            self.use_paddle_cluster = True
        else:
            self.search_fn = self._native_neighbor_search
            self.use_paddle_cluster = False

    def _paddle_cluster_search(self, data, queries, radi, device="cpu"):
        """
        Perform fixed radius search using paddle_cluster.radius.

        Parameters
        ----------
        data : paddle.Tensor of shape [n, d]
            Search space of possible neighbors.
        queries : paddle.Tensor of shape [m, d]
            Points for which to find neighbors.
        radius : float
            Radius of each ball: B(queries[j], radius).

        Returns
        -------
        dict
            Keys: 'neighbors_index', 'neighbors_row_splits' in CSR format.
        """
        data = data.to(device=device)
        queries = queries.to(device=device)
        row, col = paddle_cluster_radius(data, queries, radi)
        num_queries = queries.shape[0]
        neighbors_index = col.long()
        counts = paddle.bincount(x=row, minlength=num_queries)
        neighbors_row_splits = counts.cumsum(0)
        zero_tensor = paddle.tensor([0], device=row.place, dtype=paddle.long)
        neighbors_row_splits = paddle.cat([zero_tensor, neighbors_row_splits]).long()
        return {
            "neighbors_index": neighbors_index,
            "neighbors_row_splits": neighbors_row_splits,
        }

    def _native_neighbor_search(self, data, queries, radi, device="cpu"):
        data_comp = data.to(device=device)
        queries_comp = queries.to(device=device)
        return native_neighbor_search(data_comp, queries_comp, radi)

    def forward(self, data, queries, radi, device="cpu"):
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : paddle.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : paddle.Tensor of shape [m, d]
            Points for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)
        devices : str
            Device to run the search on. Default is "cpu".

        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: paddle.Tensor with dtype=paddle.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and paddle_cluster
                    implementations can differ by a permutation of the
                    neighbors for every point.
                neighbors_row_splits: paddle.Tensor of shape [m+1] with dtype=paddle.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        if self.use_paddle_cluster:
            result_dict = self._paddle_cluster_search(
                data, queries, radi, device=device
            )
        else:
            result_dict = self._native_neighbor_search(
                data, queries, radi, device=device
            )
        return {
            "neighbors_index": result_dict["neighbors_index"].cpu(),
            "neighbors_row_splits": result_dict["neighbors_row_splits"].cpu(),
        }


def native_neighbor_search(
    data: paddle.Tensor, queries: paddle.Tensor, radius: paddle.Tensor
):
    """
    Native Pypaddle implementation of a neighborhood search
    between two arbitrary coordinate meshes.

    Parameters
    -----------

    data : paddle.Tensor
        Vector of data points from which to find neighbors. Shape: (num_data, D)
    queries : paddle.Tensor
        Centers of neighborhoods. Shape: (num_queries, D)
    radius : paddle.Tensor or float
        Size of each neighborhood. If tensor, should be of shape (num_queries,)
    """
    if isinstance(radius, paddle.Tensor):
        if radius.dim() != 1 or radius.size(0) != queries.size(0):
            raise ValueError(
                "If radius is a tensor, it must be one-dimensional and match the number of queries."
            )
        radius = radius.view(-1, 1)
    else:
        radius = paddle.tensor(radius, device=queries.place).view(1, 1)
    with paddle.no_grad():
        dists = paddle.cdist(x=queries, y=data).to(queries.place)
        in_nbr = paddle.where(dists <= radius, 1.0, 0.0)
        nbr_indices = in_nbr.nonzero()[:, 1:].reshape(-1)
        nbrhd_sizes = paddle.cumsum(paddle.sum(in_nbr, dim=1), dim=0)
        splits = paddle.cat((paddle.tensor([0.0]).to(queries.place), nbrhd_sizes))
        nbr_dict = {}
        nbr_dict["neighbors_index"] = nbr_indices.long().to(queries.place)
        nbr_dict["neighbors_row_splits"] = splits.long()
        del dists, in_nbr, nbr_indices, nbrhd_sizes, splits
        paddle.cuda.empty_cache()
    return nbr_dict


def segment_csr(
    src: paddle.Tensor,
    indptr: paddle.Tensor,
    reduce: Literal["mean", "sum"],
    use_scatter=True,
):
    """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    If use_scatter is set to False or paddle_scatter is not
    properly built, segment_csr falls back to a naive Pypaddle implementation

    Note: the native version is mainly intended for running tests on
    CPU-only GitHub CI runners to get around a versioning issue.
    paddle_scatter should be installed and built if possible.

    Parameters
    ----------
    src : paddle.Tensor
        tensor of features for each point
    indptr : paddle.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    """
    if not use_scatter and reduce not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")
    if importlib.util.find_spec("paddle_scatter") is not None and use_scatter:
        """only import paddle_scatter when cuda is available"""
        import paddle_scatter.segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduce)
    else:
        if use_scatter:
            print(
                "Warning: use_scatter is True but paddle_scatter is not properly built.                   Defaulting to naive Pypaddle implementation"
            )
        if src.ndim == 3:
            batched = True
            point_dim = 1
        else:
            batched = False
            point_dim = 0
        output_shape = list(src.shape)
        n_out = indptr.shape[point_dim] - 1
        output_shape[point_dim] = n_out
        out = paddle.zeros(output_shape, device=src.place)
        for i in range(n_out):
            if batched:
                from_idx = slice(None), slice(indptr[0, i], indptr[0, i + 1])
                ein_str = "bio->bo"
                start = indptr[0, i]
                n_nbrs = indptr[0, i + 1] - start
                to_idx = slice(None), i
            else:
                from_idx = slice(indptr[i], indptr[i + 1])
                ein_str = "io->o"
                start = indptr[i]
                n_nbrs = indptr[i + 1] - start
                to_idx = i
            src_from = src[from_idx]
            if n_nbrs > 0:
                to_reduce = paddle.einsum(ein_str, src_from)
                if reduce == "mean":
                    to_reduce /= n_nbrs
                out[to_idx] += to_reduce
        return out


class Activation(nn.Layer):
    """
    Parameters:
    -----------
        x: paddle.FloatTensor
            input tensor
    Returns:
    --------
        y: paddle.FloatTensor
            output tensor, same shape as the input tensor, since it's element-wise operation
    """

    def __init__(self, activation: str):
        super().__init__()
        activation = activation.lower()
        if activation in ["sigmoid", "tanh"]:
            self.activation_fn = getattr(paddle, activation)
        elif activation == "swish":
            self.beta = nn.Parameter(paddle.ones(1), requires_grad=True)
        elif activation == "identity":
            self.activation_fn = lambda x: x
        else:
            self.activation_fn = getattr(F, activation)
        self.activation = activation

    def forward(self, x):
        if self.activation == "swish":
            return x * nn.functional.sigmoid(self.beta * x)
        elif self.activation == "gelu":
            return x * nn.functional.sigmoid(1.702 * x)
        elif self.activation == "mish":
            return x * nn.functional.tanh(x=nn.functional.softplus(x=x))
        else:
            return self.activation_fn(x)
