from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import paddle_geometric as pyg

# from layers.utils.scale import rescale
from layers.geoembed import GeometricEmbedding
from layers.integral_transform import IntegralTransform
from layers.mlp import ChannelMLP
from layers.mlp import LinearChannelMLP
from omegaconf import ListConfig
from omegaconf import OmegaConf
from paddle import nn
from paddle_geometric.nn import knn as pyg_knn
from paddle_geometric.nn import radius as pyg_radius
from paddle_geometric.utils import coalesce
from paddle_geometric.utils import dropout_edge


def rescale_new(x: paddle.Tensor, lims=(-1, 1), phys_domain=([-1, -1, -1], [1, 1, 1])):
    min_vals = paddle.tensor(phys_domain).min()
    max_vals = paddle.tensor(phys_domain).max()

    rescaled = ((x - min_vals) / (max_vals - min_vals)) * (lims[1] - lims[0]) + lims[0]

    return rescaled


def normalize(x: paddle.Tensor, mean=None, std=None, return_mean_std: bool = False):
    """
    Parameters
    ----------
    x: paddle.Tensor
        ND tensor
    mean: Optional[float]
        mean of the data
    std: Optional[float]
        standard deviation of the data

    Returns
    -------
    x_normalized: paddle.Tensor
        1D tensor
    """
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / std


def rescale(x: paddle.Tensor, lims=(-1, 1)) -> paddle.Tensor:
    """
    Parameters
    ----------
    x: paddle.Tensor
        ND tensor

    Returns
    -------
    x_normalized: paddle.Tensor
        ND tensor
    """
    return (x - x.min()) / (x.max() - x.min()) * (lims[1] - lims[0]) + lims[0]


############
# MAGNO Config
############

MAGNOConfig = OmegaConf.create(
    {
        "use_gno": True,
        "gno_coord_dim": 3,
        "neighbor_strategy": "bidirectional",
        "projection_channels": 256,
        "in_gno_channel_mlp_hidden_layers": [64, 64, 64],
        "out_gno_channel_mlp_hidden_layers": [64, 64],
        "lifting_channels": 32,
        "gno_radius": 0.033,
        "gno_use_torch_cluster": True,
        "attention_type": "cosine",
        "use_geoembed": [True, False],
        "embedding_method": "statistical",
        "encoder_feature_attr": ["pos", "c"],
        "mlp_type": "channel",
        "precompute_edges": True,
        "asynchronous_graph_building": False,
        "in_gno_transform_type": "linear",
        "out_gno_transform_type": "linear",
        "scales": [1.0],
        "use_scale_weights": False,
        "use_graph_cache": True,
        "gno_use_torch_scatter": True,
        "node_embedding": False,
        "use_attn": None,
        "pooling": "max",
        "sampling_strategy": None,
        "max_neighbors": None,
        "sample_ratio": None,
        "k_neighbors": 1,
    }
)


def parse_neighbor_strategy(
    neighbor_strategy: Union[str, List[str]]
) -> Tuple[str, str]:
    """
    Parse neighbor_strategy into encoder and decoder strategies.

    Args:
        neighbor_strategy: Either a string (same for both) or [encoder_strategy, decoder_strategy]

    Returns:
        Tuple[str, str]: (encoder_strategy, decoder_strategy)

    Examples:
        parse_neighbor_strategy('radius') -> ('radius', 'radius')
        parse_neighbor_strategy(['knn', 'bidirectional']) -> ('knn', 'bidirectional')
    """
    if isinstance(neighbor_strategy, str):
        return neighbor_strategy, neighbor_strategy
    elif isinstance(neighbor_strategy, list) and len(neighbor_strategy) == 2:
        return neighbor_strategy[0], neighbor_strategy[1]
    else:
        raise ValueError(
            f"neighbor_strategy must be str or list of length 2, got {neighbor_strategy}"
        )


def parse_geoembed_strategy(use_geoembed: Union[bool, List[bool]]) -> Tuple[bool, bool]:
    """
    Parse use_geoembed into encoder and decoder settings.

    Args:
        use_geoembed: Either a bool (same for both) or [encoder_setting, decoder_setting]

    Returns:
        Tuple[bool, bool]: (use_geoembed_encoder, use_geoembed_decoder)

    Examples:
        parse_geoembed_strategy(True) -> (True, True)
        parse_geoembed_strategy(False) -> (False, False)
        parse_geoembed_strategy([True, False]) -> (True, False)
        parse_geoembed_strategy([False, True]) -> (False, True)
    """
    if isinstance(use_geoembed, bool):
        return use_geoembed, use_geoembed
    elif isinstance(use_geoembed, (list, ListConfig)) and len(use_geoembed) == 2:
        return use_geoembed[0], use_geoembed[1]
    else:
        raise ValueError(
            f"use_geoembed must be bool or list of length 2, got {use_geoembed}"
            f"{type(use_geoembed)}"
        )


def get_neighbor_strategy(
    neighbor_strategy: str,
    phys_pos: paddle.Tensor,
    batch_idx_phys: paddle.Tensor,
    latent_tokens_pos: paddle.Tensor,
    batch_idx_latent: paddle.Tensor,
    radius: float,
    k_neighbors: int = 1,
    is_decoder: bool = False,
):
    """
    Get the neighbor strategy based on the provided string.

    For ENCODER (is_decoder=False):
        - latent tokens are query points, fetching info from physical points
        - knn: each physical point connects to k nearest latent tokens (phys->latent)
        - radius: latent tokens as centers, physical points within radius connect (phys->latent)
        - bidirectional: merge knn and radius strategies

    For DECODER (is_decoder=True):
        - physical points are query points, fetching info from latent tokens
        - knn: each physical point connects to k nearest latent tokens (latent->phys)
        - radius: physical points as centers, latent tokens within radius connect (latent->phys)
        - bidirectional: merge knn and radius strategies
        - reverse: direct inverse of encoder graph

    Args:
        neighbor_strategy (str): Strategy to use ['knn', 'radius', 'bidirectional', 'reverse']
        phys_pos (Tensor): Physical positions [N_phys, 3]
        batch_idx_phys (Tensor): Batch indices for physical positions [N_phys]
        latent_tokens_pos (Tensor): Latent token positions [N_latent, 3]
        batch_idx_latent (Tensor): Batch indices for latent token positions [N_latent]
        radius (float): Radius for neighbor finding
        k_neighbors (int): Number of nearest neighbors for knn strategy
        is_decoder (bool): Whether this is for decoder graph construction
    Returns:
        edge_index (Tensor): Edge index [2, N_edges], format depends on encoder/decoder
    """
    if is_decoder:
        return _get_decoder_strategy(
            neighbor_strategy,
            phys_pos,
            batch_idx_phys,
            latent_tokens_pos,
            batch_idx_latent,
            radius,
            k_neighbors,
        )
    else:
        return _get_encoder_strategy(
            neighbor_strategy,
            phys_pos,
            batch_idx_phys,
            latent_tokens_pos,
            batch_idx_latent,
            radius,
            k_neighbors,
        )


def _get_encoder_strategy(
    neighbor_strategy: str,
    phys_pos: paddle.Tensor,
    batch_idx_phys: paddle.Tensor,
    latent_tokens_pos: paddle.Tensor,
    batch_idx_latent: paddle.Tensor,
    radius: float,
    k_neighbors: int,
):
    """
    Encoder strategies: latent tokens as query points, fetching from physical points
    Edge direction: physical -> latent (info flows from physical to latent)
    """
    device = phys_pos.place
    edge_index_knn = None
    edge_index_radius = None
    if neighbor_strategy in ["knn", "bidirectional"]:
        edge_index_knn = pyg_knn(
            x=latent_tokens_pos,
            y=phys_pos,
            k=k_neighbors,
            batch_x=batch_idx_latent,
            batch_y=batch_idx_phys,
        )
    if neighbor_strategy in ["radius", "bidirectional"]:
        edge_index_radius_raw = pyg_radius(
            x=phys_pos,
            y=latent_tokens_pos,
            r=radius,
            batch_x=batch_idx_phys,
            batch_y=batch_idx_latent,
        )
        edge_index_radius = edge_index_radius_raw.flip(axis=0)
    if neighbor_strategy == "knn":
        return (
            edge_index_knn
            if edge_index_knn is not None
            else paddle.empty((2, 0), dtype=paddle.long, device=device)
        )
    elif neighbor_strategy == "radius":
        return (
            edge_index_radius
            if edge_index_radius is not None
            else paddle.empty((2, 0), dtype=paddle.long, device=device)
        )
    elif neighbor_strategy == "bidirectional":
        edges = []
        if edge_index_knn is not None:
            edges.append(edge_index_knn)
        if edge_index_radius is not None:
            edges.append(edge_index_radius)
        if len(edges) == 0:
            return paddle.empty((2, 0), dtype=paddle.long, device=device)
        elif len(edges) == 1:
            return edges[0]
        else:
            combined = paddle.concat(edges, dim=1)
            return coalesce(combined)
    else:
        raise ValueError(f"Unknown encoder strategy: {neighbor_strategy}")


def _get_decoder_strategy(
    neighbor_strategy: str,
    phys_pos: paddle.Tensor,
    batch_idx_phys: paddle.Tensor,
    latent_tokens_pos: paddle.Tensor,
    batch_idx_latent: paddle.Tensor,
    radius: float,
    k_neighbors: int,
):
    """
    Decoder strategies: physical points as query points, fetching from latent tokens
    Edge direction: latent -> physical (info flows from latent to physical)
    """
    device = phys_pos.place
    edge_index_knn = None
    edge_index_radius = None
    if neighbor_strategy in ["knn", "bidirectional"]:
        edge_index_knn_raw = pyg_knn(
            x=latent_tokens_pos,
            y=phys_pos,
            k=k_neighbors,
            batch_x=batch_idx_latent,
            batch_y=batch_idx_phys,
        )
        edge_index_knn = edge_index_knn_raw.flip(axis=0)
    if neighbor_strategy in ["radius", "bidirectional"]:
        edge_index_radius_raw = pyg_radius(
            x=latent_tokens_pos,
            y=phys_pos,
            r=radius,
            batch_x=batch_idx_latent,
            batch_y=batch_idx_phys,
        )
        edge_index_radius = edge_index_radius_raw.flip(axis=0)
    if neighbor_strategy == "reverse":
        encoder_edges = _get_encoder_strategy(
            "bidirectional",
            phys_pos,
            batch_idx_phys,
            latent_tokens_pos,
            batch_idx_latent,
            radius,
            k_neighbors,
        )
        return encoder_edges.flip(axis=0)
    if neighbor_strategy == "knn":
        return (
            edge_index_knn
            if edge_index_knn is not None
            else paddle.empty((2, 0), dtype=paddle.long, device=device)
        )
    elif neighbor_strategy == "radius":
        return (
            edge_index_radius
            if edge_index_radius is not None
            else paddle.empty((2, 0), dtype=paddle.long, device=device)
        )
    elif neighbor_strategy == "bidirectional":
        edges = []
        if edge_index_knn is not None:
            edges.append(edge_index_knn)
        if edge_index_radius is not None:
            edges.append(edge_index_radius)
        if len(edges) == 0:
            return paddle.empty((2, 0), dtype=paddle.long, device=device)
        elif len(edges) == 1:
            return edges[0]
        else:
            combined = paddle.concat(edges, dim=1)
            return coalesce(combined)
    else:
        raise ValueError(f"Unknown decoder strategy: {neighbor_strategy}")


def apply_neighbor_sampling(
    edge_index: paddle.Tensor,
    num_query_nodes: int,
    device: paddle.device,
    sampling_strategy: Optional[str] = None,
    max_neighbors: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    training: bool = True,
) -> paddle.Tensor:
    """
    Applies neighbor sampling based on the configured strategy.

    Args:
        edge_index (Tensor): Edge index [2, num_edges]
        num_query_nodes (int): Number of query nodes
        device (paddle.device): Device for tensor operations
        sampling_strategy (str, optional): Sampling strategy ['max_neighbors', 'ratio']
        max_neighbors (int, optional): Maximum number of neighbors per node
        sample_ratio (float, optional): Ratio of edges to keep
        training (bool): Whether model is in training mode

    Returns:
        Tensor: Sampled edge index [2, num_sampled_edges]
    """
    if sampling_strategy is None:
        return edge_index
    num_total_original_edges = edge_index.shape[1]
    if num_query_nodes == 0 or num_total_original_edges == 0:
        return edge_index
    if sampling_strategy == "max_neighbors":
        if max_neighbors is None:
            raise ValueError(
                "max_neighbors must be provided when using 'max_neighbors' sampling strategy"
            )
        dest_nodes = edge_index[1]
        counts = paddle.bincount(x=dest_nodes, minlength=num_query_nodes)
        needs_sampling_mask = counts > max_neighbors
        if not paddle.any(needs_sampling_mask):
            return edge_index
        keep_mask = paddle.ones(
            num_total_original_edges, dtype=paddle.bool, device=device
        )
        queries_to_sample_idx = paddle.where(needs_sampling_mask)[0]
        for i in queries_to_sample_idx:
            node_edge_mask = dest_nodes == i
            node_edge_indices = paddle.where(node_edge_mask)[0]
            num_node_edges = len(node_edge_indices)
            perm = paddle.randperm(num_node_edges, device=device)[:max_neighbors]
            edges_to_keep_for_node = node_edge_indices[perm]
            node_keep_mask = paddle.zeros_like(node_edge_mask)
            node_keep_mask[edges_to_keep_for_node] = True
            keep_mask[node_edge_mask] = node_keep_mask[node_edge_mask]
        sampled_edge_index = edge_index[:, keep_mask]
        return sampled_edge_index
    elif sampling_strategy == "ratio":
        if sample_ratio is None:
            raise ValueError(
                "sample_ratio must be provided when using 'ratio' sampling strategy"
            )
        if sample_ratio >= 1.0:
            return edge_index
        p_drop = 1.0 - sample_ratio
        sampled_edge_index, _ = dropout_edge(
            edge_index, p=p_drop, force_undirected=False, training=training
        )
        return sampled_edge_index
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")


class MAGNOEncoder(paddle.nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        gno_config: OmegaConf = MAGNOConfig,
    ):
        super().__init__()
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.lifting_channels = gno_config.lifting_channels
        self.coord_dim = gno_config.gno_coord_dim
        self.feature_attr_name = gno_config.encoder_feature_attr
        self.precompute_edges = gno_config.precompute_edges
        self.mlp_type = gno_config.mlp_type

        # --- Store Neighbor Finding Strategy ---
        self.encoder_strategy, self.decoder_strategy = parse_neighbor_strategy(
            gno_config.neighbor_strategy
        )
        self.k_neighbors = gno_config.k_neighbors

        # --- Store Sampling Strategy ---
        self.sampling_strategy = gno_config.sampling_strategy
        self.max_neighbors = gno_config.max_neighbors
        self.sample_ratio = gno_config.sample_ratio
        if self.sampling_strategy == "max_neighbors":
            print(
                "Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'."
            )
        # --- Init GNO Layer ---

        ## --- Calculate MLP input dimension ---
        self.use_gno = gno_config.use_gno
        if self.use_gno:
            in_kernel_in_dim = self.coord_dim * 2
            if gno_config.in_gno_transform_type in [
                "nonlinear",
                "nonlinear_kernelonly",
            ]:
                in_kernel_in_dim += in_channels

            in_gno_channel_mlp_hidden_layers = (
                gno_config.in_gno_channel_mlp_hidden_layers.copy()
            )
            in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
            in_gno_channel_mlp_hidden_layers.append(
                self.lifting_channels
            )  # Kernel MLP output dim

            self.gno = IntegralTransform(
                channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
                transform_type=gno_config.in_gno_transform_type,
                # use_torch_scatter determined globally now
                use_attn=gno_config.use_attn,
                coord_dim=self.coord_dim,  # Pass coord_dim if attn used
                attention_type=gno_config.attention_type,
            )

            ## --- Init Lifting MLP ---
            if gno_config.mlp_type == "linear":
                self.lifting = LinearChannelMLP(
                    layers=[in_channels, self.lifting_channels]
                )
            else:
                self.lifting = ChannelMLP(
                    in_channels=in_channels,
                    out_channels=self.lifting_channels,  # Output matches GNO kernel output
                    n_layers=1,
                )
        else:
            self.gno = None
            self.lifting = None
            if in_channels > 0:
                print(
                    "Warning: MAGNOEncoder has input_channels > 0 but use_gno=False. Input features (batch.x) will be ignored by the encoder path."
                )

        # --- Init GeoEmbed （optional) ---
        use_geoembed_encoder, use_geoembed_decoder = parse_geoembed_strategy(
            gno_config.use_geoembed
        )
        self.use_geoembed = use_geoembed_encoder  # Use encoder-specific setting
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=self.lifting_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling,
            )
            if gno_config.mlp_type == "linear":
                self.recovery = LinearChannelMLP(
                    layers=[2 * self.lifting_channels, self.lifting_channels]
                )
            else:
                self.recovery = ChannelMLP(
                    in_channels=2 * self.lifting_channels,
                    out_channels=self.lifting_channels,
                    n_layers=1,
                )

        # --- Init Scale Weighting (optional) ---
        self.use_scale_weights = gno_config.use_scale_weights
        if self.use_scale_weights:
            # Weighting based on latent token positions
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16), nn.ReLU(), nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(
        self,
        batch: "pyg.data.Batch",
        latent_tokens_pos: paddle.Tensor,
        latent_tokens_batch_idx: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Args:
            batch (Batch): PyG batch object (pos, x, batch for physical).
            latent_tokens_pos (Tensor): Latent token coordinates [TotalLatentNodes, D].
            latent_tokens_batch_idx (Tensor): Batch index for latent tokens [TotalLatentNodes].
        """
        phys_pos = paddle.from_dlpack(batch.pos)
        batch_idx_phys = paddle.from_dlpack(batch.batch)
        device = phys_pos.place
        num_graphs = batch.num_graphs
        num_latent_tokens_per_graph = latent_tokens_pos.shape[0] // num_graphs
        if isinstance(self.feature_attr_name, (list, ListConfig)):
            phys_feats = []
            for attr_name in self.feature_attr_name:
                feat = getattr(batch, attr_name, None)
                if feat is None:
                    if self.use_gno:
                        raise AttributeError(
                            f"MAGNOEncoder requires feature attribute '{attr_name}' but it was not found in the batch."
                        )
                else:
                    phys_feats.append(paddle.from_dlpack(feat))
            phys_feat = paddle.concat(phys_feats, dim=-1) if phys_feats else None
        else:
            phys_feat = getattr(batch, self.feature_attr_name, None)
            if phys_feat is None:
                if self.use_gno:
                    raise AttributeError(
                        f"MAGNOEncoder requires feature attribute '{self.feature_attr_name}' but it was not found in the batch."
                    )
        encoded_scales = []
        for scale_idx, scale in enumerate(self.scales):
            scaled_radius = self.gno_radius * scale
            if self.precompute_edges:
                edge_index_attr = f"encoder_edge_index_s{scale_idx}"
                if not hasattr(batch, edge_index_attr):
                    raise AttributeError(
                        f"Batch object missing pre-computed '{edge_index_attr}'"
                    )
                edge_index = paddle.from_dlpack(getattr(batch, edge_index_attr)).to(
                    device
                )
                neighbor_counts = None
            else:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy=self.encoder_strategy,
                    phys_pos=phys_pos,
                    batch_idx_phys=batch_idx_phys,
                    latent_tokens_pos=latent_tokens_pos,
                    batch_idx_latent=latent_tokens_batch_idx,
                    radius=scaled_radius,
                    k_neighbors=self.k_neighbors,
                    is_decoder=False,
                )
                neighbor_counts = None
            edge_index = apply_neighbor_sampling(
                edge_index=edge_index,
                num_query_nodes=latent_tokens_pos.shape[0],
                device=device,
                sampling_strategy=self.sampling_strategy,
                max_neighbors=self.max_neighbors,
                sample_ratio=self.sample_ratio,
                training=self.training,
            )
            if self.use_gno:
                if self.mlp_type == "linear":
                    phys_feat_lifted = self.lifting(phys_feat)
                else:
                    phys_feat_lifted = self.lifting(
                        phys_feat.transpose(0, 1)
                    ).transpose(0, 1)
                encoded_gno = self.gno(
                    y_pos=phys_pos,
                    x_pos=latent_tokens_pos,
                    edge_index=edge_index,
                    f_y=phys_feat_lifted,
                    batch_y=batch_idx_phys,
                    batch_x=latent_tokens_batch_idx,
                )
            else:
                encoded_gno = None
            if self.use_geoembed:
                geo_embedding = self.geoembed(
                    source_pos=phys_pos,
                    query_pos=latent_tokens_pos,
                    edge_index=edge_index,
                    batch_source=batch_idx_phys,
                    batch_query=latent_tokens_batch_idx,
                    neighbors_counts=neighbor_counts,
                )
            else:
                geo_embedding = None
            if self.use_gno and self.use_geoembed:
                combined = paddle.concat([encoded_gno, geo_embedding], dim=-1)
                if self.mlp_type == "linear":
                    encoded_unpatched = self.recovery(combined)
                else:
                    encoded_unpatched = self.recovery(combined.permute(1, 0)).permute(
                        1, 0
                    )
            elif self.use_gno:
                encoded_unpatched = encoded_gno
            elif self.use_geoembed:
                encoded_unpatched = geo_embedding
            else:
                raise ValueError(
                    "GNO and GeoEmbed are both disabled. No encoding will be performed."
                )
            encoded_scales.append(encoded_unpatched)
        if len(encoded_scales) == 1:
            encoded_data = encoded_scales[0]
        else:
            encoded_stack = paddle.stack(encoded_scales, dim=0)
            if self.use_scale_weights:
                scale_w = self.scale_weighting(latent_tokens_pos)
                scale_w = self.scale_weight_activation(scale_w)
                weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                encoded_data = (encoded_stack * weights_reshaped).sum(dim=0)
            else:
                encoded_data = encoded_stack.sum(dim=0)
        encoded_data = encoded_data.view(
            num_graphs, num_latent_tokens_per_graph, self.lifting_channels
        )
        return encoded_data


class MAGNODecoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, gno_config: MAGNOConfig):
        super().__init__()
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.coord_dim = gno_config.gno_coord_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        use_geoembed_encoder, use_geoembed_decoder = parse_geoembed_strategy(
            gno_config.use_geoembed
        )
        self.use_geoembed = use_geoembed_decoder
        self.use_scale_weights = gno_config.use_scale_weights
        self.precompute_edges = gno_config.precompute_edges
        self.mlp_type = gno_config.mlp_type
        self.encoder_strategy, self.decoder_strategy = parse_neighbor_strategy(
            gno_config.neighbor_strategy
        )
        self.k_neighbors = gno_config.k_neighbors
        self.sampling_strategy = gno_config.sampling_strategy
        self.max_neighbors = gno_config.max_neighbors
        self.sample_ratio = gno_config.sample_ratio
        if self.sampling_strategy == "max_neighbors":
            print(
                "Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'."
            )
        out_kernel_in_dim = self.coord_dim * 2
        if gno_config.out_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            out_kernel_in_dim += self.in_channels
        out_gno_channel_mlp_hidden_layers = (
            gno_config.out_gno_channel_mlp_hidden_layers.copy()
        )
        out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_channel_mlp_hidden_layers.append(self.in_channels)
        self.gno = IntegralTransform(
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.out_gno_transform_type,
            use_attn=gno_config.use_attn,
            coord_dim=self.coord_dim,
            attention_type=gno_config.attention_type,
        )
        if gno_config.mlp_type == "linear":
            self.projection = LinearChannelMLP(
                layers=[in_channels, gno_config.projection_channels, out_channels]
            )
        else:
            self.projection = ChannelMLP(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=gno_config.projection_channels,
                n_layers=2,
                n_dim=1,
            )
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=self.in_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling,
            )
            if gno_config.mlp_type == "linear":
                self.recovery = LinearChannelMLP(layers=[2 * in_channels, in_channels])
            else:
                self.recovery = ChannelMLP(
                    in_channels=2 * in_channels, out_channels=in_channels, n_layers=1
                )
        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.scale_weighting = paddle.nn.Sequential(
                paddle.nn.Linear(in_features=self.coord_dim, out_features=16),
                paddle.nn.ReLU(),
                paddle.nn.Linear(in_features=16, out_features=self.num_scales),
            )
            self.scale_weight_activation = paddle.nn.Softmax(axis=-1)

    def forward(
        self,
        rndata_flat: paddle.Tensor,
        phys_pos_query: paddle.Tensor,
        batch_idx_phys_query: paddle.Tensor,
        latent_tokens_pos: paddle.Tensor,
        latent_tokens_batch_idx: paddle.Tensor,
        batch: "pyg.data.Batch" = None,
    ) -> paddle.Tensor:
        """
        Args:
            rndata_flat (Tensor): Latent features (source) [TotalLatentNodes, C_in].
            phys_pos_query (Tensor): Physical/Query coordinates (dest) [TotalQueryNodes, D].
            batch_idx_phys_query (Tensor): Batch index for physical/query nodes [TotalQueryNodes].
            latent_tokens_pos (Tensor): Latent token coordinates (source) [TotalLatentNodes, D].
            latent_tokens_batch_idx (Tensor): Batch index for latent tokens (source) [TotalLatentNodes].
            batch (Batch): Optional PyG batch object for precomputed edges. [TotalQueryNodes, C_out]
        """
        assert isinstance(rndata_flat, paddle.Tensor)
        assert isinstance(phys_pos_query, paddle.Tensor)
        assert isinstance(batch_idx_phys_query, paddle.Tensor)
        assert isinstance(latent_tokens_pos, paddle.Tensor)
        assert isinstance(latent_tokens_batch_idx, paddle.Tensor)
        device = rndata_flat.place
        decoded_scales = []
        for scale_idx, scale in enumerate(self.scales):
            scaled_radius = self.gno_radius * scale
            if self.precompute_edges:
                edge_index_attr = f"decoder_edge_index_s{scale_idx}"
                if not hasattr(batch, edge_index_attr):
                    raise AttributeError(
                        f"Batch object missing pre-computed '{edge_index_attr}'"
                    )
                edge_index = paddle.from_dlpack(getattr(batch, edge_index_attr)).to(
                    device
                )
                neighbor_counts = None
            else:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy=self.decoder_strategy,
                    phys_pos=phys_pos_query,
                    batch_idx_phys=batch_idx_phys_query,
                    latent_tokens_pos=latent_tokens_pos,
                    batch_idx_latent=latent_tokens_batch_idx,
                    radius=scaled_radius,
                    k_neighbors=self.k_neighbors,
                    is_decoder=True,
                )
                neighbor_counts = None
            edge_index = apply_neighbor_sampling(
                edge_index=edge_index,
                num_query_nodes=phys_pos_query.shape[0],
                device=device,
                sampling_strategy=self.sampling_strategy,
                max_neighbors=self.max_neighbors,
                sample_ratio=self.sample_ratio,
                training=self.training,
            )
            decoded_unpatched = self.gno(
                y_pos=latent_tokens_pos,
                x_pos=phys_pos_query,
                edge_index=edge_index,
                f_y=rndata_flat,
                batch_y=latent_tokens_batch_idx,
                batch_x=batch_idx_phys_query,
            )
            if self.use_geoembed:
                geoembedding = self.geoembed(
                    source_pos=latent_tokens_pos,
                    query_pos=phys_pos_query,
                    edge_index=edge_index,
                    batch_source=latent_tokens_batch_idx,
                    batch_query=batch_idx_phys_query,
                    neighbors_counts=neighbor_counts,
                )
                combined = paddle.concat([decoded_unpatched, geoembedding], dim=-1)
                if self.mlp_type == "linear":
                    decoded_unpatched = self.recovery(combined)
                else:
                    decoded_unpatched = self.recovery(combined.permute(1, 0)).permute(
                        1, 0
                    )
            decoded_scales.append(decoded_unpatched)
        if len(decoded_scales) == 1:
            decoded_data = decoded_scales[0]
        else:
            decoded_stack = paddle.stack(decoded_scales, dim=0)
            if self.use_scale_weights:
                scale_w = self.scale_weighting(phys_pos_query)
                scale_w = self.scale_weight_activation(scale_w)
                weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                decoded_data = (decoded_stack * weights_reshaped).sum(dim=0)
            else:
                decoded_data = decoded_stack.sum(dim=0)
        if self.mlp_type == "linear":
            decoded_data = self.projection(decoded_data)
        else:
            decoded_data = decoded_data.permute(1, 0)
            decoded_data = self.projection(decoded_data).permute(1, 0)
        return decoded_data
