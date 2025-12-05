from __future__ import annotations

from typing import Optional

import paddle

try:
    import paddle_scatter

    if hasattr(paddle_scatter, "scatter"):
        scatter = paddle_scatter.scatter
        HAS_PADDLE_SCATTER = True
    else:
        HAS_PADDLE_SCATTER = False
except ImportError:
    HAS_PADDLE_SCATTER = False
if not HAS_PADDLE_SCATTER:
    print(
        "Warning: torch_scatter.scatter not found. Using native PyTorch fallbacks (potentially slower)."
    )
    from .utils.scatter_native import scatter_native

    scatter = scatter_native


class GeometricEmbedding(paddle.nn.Layer):
    def __init__(
        self, input_dim, output_dim, method="statistical", pooling="max", **kwargs
    ):
        super(GeometricEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method.lower()
        self.pooling = pooling.lower()
        self.kwargs = kwargs
        if self.pooling not in ["max", "mean"]:
            raise ValueError(
                f"Unsupported pooling method: {self.pooling}. Supported methods: 'max', 'mean'."
            )
        if self.method == "statistical":
            self.mlp = paddle.nn.Sequential(
                paddle.nn.Linear(
                    in_features=self._get_stat_feature_dim(), out_features=64
                ),
                paddle.nn.ReLU(),
                paddle.nn.Linear(in_features=64, out_features=output_dim),
            )
        elif self.method == "pointnet":
            self.pointnet_mlp = paddle.nn.Sequential(
                paddle.nn.Linear(in_features=input_dim, out_features=32),
                paddle.nn.ReLU(),
                paddle.nn.Linear(in_features=32, out_features=32),
                paddle.nn.ReLU(),
            )
            self.fc = paddle.nn.Sequential(
                paddle.nn.Linear(in_features=32, out_features=output_dim)
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forward(
        self,
        source_pos: paddle.Tensor,
        query_pos: paddle.Tensor,
        edge_index: paddle.Tensor,
        batch_source: Optional[paddle.Tensor] = None,
        batch_query: Optional[paddle.Tensor] = None,
        neighbors_counts: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Compute geometric embeddings using PyG batch format.

        Args:
            source_pos (Tensor): Coords of source nodes providing geometry [TotalSourceNodes, D].
            query_pos (Tensor): Coords of query nodes for which embeddings are computed [TotalQueryNodes, D].
            edge_index (Tensor): Bipartite edges [2, NumEdges], where edge_index[1] indexes query_pos,
                                 and edge_index[0] indexes source_pos.
            batch_source (Tensor, optional): Batch index for source nodes.
            batch_query (Tensor, optional): Batch index for query nodes.
            neighbors_counts (Tensor, optional): Number of neighbors for each query node.

        Returns:
            Tensor: Geometric embeddings for query nodes [TotalQueryNodes, output_dim].
        """
        if self.method == "statistical":
            geo_features = self._compute_statistical_features_pyg(
                source_pos, query_pos, edge_index, neighbors_counts
            )
            return self.mlp(geo_features)
        elif self.method == "pointnet":
            geo_features = self._compute_pointnet_features_pyg(
                source_pos, query_pos, edge_index
            )
            return geo_features
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _get_stat_feature_dim(self):
        return 3 + 2 * self.input_dim

    def _compute_statistical_features_pyg(
        self, source_pos, query_pos, edge_index, neighbors_counts=None
    ):
        """
        Computes statistical geometric features using PyG edge_index.

        Parameters:
            source_pos (Tensor): Coords of source nodes providing geometry [TotalSourceNodes, D].
            query_pos (Tensor): Coords of query nodes for which embeddings are computed [TotalQueryNodes, D].
            edge_index (Tensor): Bipartite edges [2, NumEdges], where edge_index[1] indexes query_pos,
                                and edge_index[0] indexes source_pos.
            neighbors_counts (Tensor, optional): Number of neighbors for each query node.

        Returns:
            geo_features_normalized (torch.FloatTensor): The normalized geometric features, shape: [TotalQueryNodes, num_features]
        """
        num_queries = query_pos.shape[0]
        num_dims = query_pos.shape[1]
        device = query_pos.place
        neighbors_index = edge_index[0].long()
        query_indices_per_neighbor = edge_index[1].long()
        if neighbors_counts is None:
            num_neighbors_per_query = paddle.bincount(
                x=query_indices_per_neighbor, minlength=num_queries
            ).to(device)
        else:
            neighbors_counts = neighbors_counts.to(device)
            num_neighbors_per_query = neighbors_counts
        N_i = num_neighbors_per_query.float()
        has_neighbors = N_i > 0
        nbr_coords = source_pos[neighbors_index]
        query_coords_per_neighbor = query_pos[query_indices_per_neighbor]
        distances = paddle.norm(nbr_coords - query_coords_per_neighbor, dim=1)
        D_avg = scatter(
            distances,
            query_indices_per_neighbor,
            dim=0,
            dim_size=num_queries,
            reduce="mean",
        )
        distances_squared = distances**2
        E_X2 = scatter(
            distances_squared,
            query_indices_per_neighbor,
            dim=0,
            dim_size=num_queries,
            reduce="mean",
        )
        E_X_squared = D_avg**2
        D_var = E_X2 - E_X_squared
        D_var = paddle.clamp(D_var, min=0.0)
        nbr_centroid = scatter(
            nbr_coords,
            query_indices_per_neighbor,
            dim=0,
            dim_size=num_queries,
            reduce="mean",
        )
        Delta = nbr_centroid - query_pos
        nbr_coords_centered = nbr_coords - nbr_centroid[query_indices_per_neighbor]
        cov_components = nbr_coords_centered.unsqueeze(
            2
        ) * nbr_coords_centered.unsqueeze(1)
        cov_sum = scatter(
            cov_components,
            query_indices_per_neighbor,
            dim=0,
            dim_size=num_queries,
            reduce="sum",
        )
        N_i_clamped = N_i.clone()
        N_i_clamped[N_i_clamped == 0] = 1.0
        cov_matrix = cov_sum / N_i_clamped.view(-1, 1, 1)
        PCA_features = paddle.zeros(num_queries, num_dims, device=device)
        if has_neighbors.any():
            cov_matrix_valid = cov_matrix[has_neighbors]
            eps = 1e-06
            eye = paddle.eye(num_dims, device=device, dtype=cov_matrix_valid.dtype)
            cov_matrix_reg = cov_matrix_valid + eps * eye.unsqueeze(0)
            try:
                eigenvalues = paddle.linalg.eigvalsh(x=cov_matrix_reg)
                eigenvalues = eigenvalues.flip(axis=[1])
                PCA_features[has_neighbors] = eigenvalues
            except Exception:
                default_eigenvals = paddle.ones(num_dims, device=device) * eps
                PCA_features[has_neighbors] = default_eigenvals.unsqueeze(0).expand(
                    has_neighbors.sum(), -1
                )
        N_i_tensor = N_i.unsqueeze(1)
        D_avg_tensor = D_avg.unsqueeze(1)
        D_var_tensor = D_var.unsqueeze(1)
        geo_features = paddle.cat(
            [N_i_tensor, D_avg_tensor, D_var_tensor, Delta, PCA_features], dim=1
        )
        geo_features[~has_neighbors] = 0.0
        feature_mean = geo_features.mean(dim=0, keepdim=True)
        feature_std = geo_features.std(axis=0, keepdim=True)
        feature_std[feature_std < 1e-06] = 1.0
        geo_features_normalized = (geo_features - feature_mean) / feature_std
        return geo_features_normalized

    def _compute_pointnet_features_pyg(self, source_pos, query_pos, edge_index):
        """Computes PointNet-style features using PyG edge_index."""
        num_query_nodes = query_pos.shape[0]
        device = query_pos.place
        geo_features = paddle.zeros(
            (num_query_nodes, self.output_dim), device=device, dtype=query_pos.dtype
        )
        if edge_index.size == 0:
            print("Warning: GeoEmbed (PointNet) received no edges.")
            return geo_features
        query_idx = edge_index[1].long()
        source_idx = edge_index[0].long()
        has_neighbors_mask = paddle.bincount(x=query_idx, minlength=num_query_nodes) > 0
        if not paddle.any(has_neighbors_mask):
            return geo_features
        nbr_coords = source_pos[source_idx]
        query_coords_per_edge = query_pos[query_idx]
        nbr_coords_centered = nbr_coords - query_coords_per_edge
        nbr_features = self.pointnet_mlp(nbr_coords_centered)
        if self.pooling == "max":
            pooled_features = scatter(
                nbr_features, query_idx, dim=0, dim_size=num_query_nodes, reduce="max"
            )
        elif self.pooling == "mean":
            pooled_features = scatter(
                nbr_features, query_idx, dim=0, dim_size=num_query_nodes, reduce="mean"
            )
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        pointnet_output = self.fc(pooled_features)
        geo_features[has_neighbors_mask] = pointnet_output[has_neighbors_mask]
        return geo_features
