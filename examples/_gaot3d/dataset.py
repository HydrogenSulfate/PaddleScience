from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from layers.magno import get_neighbor_strategy
from layers.magno import parse_neighbor_strategy
from layers.magno import rescale
from torch_geometric import data as pyg_data
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from ppsci.utils import logger

# src/trainer/utils/pyg_transforms.py


EPSILON = 1e-10


class RescalePosition(BaseTransform):
    """Rescales node positions 'pos' to a specified range (default: [-1, 1])."""

    def __init__(self, lims=(-1.0, 1.0)):
        self.lims = lims

    def forward(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            data.pos = rescale(data.pos, lims=self.lims)
        else:
            logger.warning(
                "Warning: RescalePosition transform called but data has no 'pos' attribute.",
                stacklevel=3,
            )
        return data

    def __call__(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            data.pos = rescale(data.pos, lims=self.lims)
        else:
            logger.warning(
                "Warning: RescalePosition transform called but data has no 'pos' attribute.",
                stacklevel=3,
            )
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lims={self.lims})"


class NormalizeFeatures(BaseTransform):
    """Normalizes node features 'x' and optionally 'c' using pre-computed mean and std."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        c_mean: torch.Tensor = None,
        c_std: torch.Tensor = None,
    ):
        self.mean = mean.detach()
        self.std = std.detach()

        self.c_mean = c_mean.detach() if c_mean is not None else None
        self.c_std = c_std.detach() if c_std is not None else None

    def forward(self, data: Data) -> Data:
        if hasattr(data, "x") and data.x is not None:
            mean_dev = self.mean.to(data.x.device)
            std_dev = self.std.to(data.x.device)
            data.x = (data.x - mean_dev) / (std_dev + EPSILON)  # Add epsilon for safety
        else:
            logger.warning(
                "Warning: NormalizeFeatures transform called but data has no 'x' attribute.",
                stacklevel=3,
            )

        if (
            hasattr(data, "c")
            and data.c is not None
            and self.c_mean is not None
            and self.c_std is not None
        ):
            c_mean_dev = self.c_mean.to(data.c.device)
            c_std_dev = self.c_std.to(data.c.device)
            data.c = (data.c - c_mean_dev) / (
                c_std_dev + EPSILON
            )  # Add epsilon for safety

        return data

    def __call__(self, data: Data) -> Data:
        if hasattr(data, "x") and data.x is not None:
            mean_dev = self.mean.to(data.x.device)
            std_dev = self.std.to(data.x.device)
            data.x = (data.x - mean_dev) / (std_dev + EPSILON)  # Add epsilon for safety
        else:
            logger.warning(
                "Warning: NormalizeFeatures transform called but data has no 'x' attribute.",
                stacklevel=3,
            )

        if (
            hasattr(data, "c")
            and data.c is not None
            and self.c_mean is not None
            and self.c_std is not None
        ):
            c_mean_dev = self.c_mean.to(data.c.device)
            c_std_dev = self.c_std.to(data.c.device)
            data.c = (data.c - c_mean_dev) / (
                c_std_dev + EPSILON
            )  # Add epsilon for safety

        return data


class EnrichedData(Data):
    """Custom Data class to handle increments for bipartite edge indices."""

    def __inc__(self, key, value, *args, **kwargs):
        """
        Specifies how attributes should be incremented when batching.
        key: The name of the attribute.
        value: The attribute's tensor value.
        """
        if key.startswith("encoder_edge_index"):
            # Encoder: edge_index[0] indexes PHYSICAL, edge_index[1] indexes LATENT
            # Increment row 0 by num_physical_nodes, row 1 by num_latent_nodes
            # Ensure 'num_latent_nodes' attribute exists in the Data object!
            return torch.tensor([[self.num_nodes], [self.num_latent_nodes]])
        elif key.startswith("decoder_edge_index"):
            # Decoder: edge_index[0] indexes LATENT, edge_index[1] indexes PHYSICAL
            # Increment row 0 by num_latent_nodes, row 1 by num_physical_nodes
            return torch.tensor([[self.num_latent_nodes], [self.num_nodes]])
        elif key.startswith("encoder_query_counts") or key.startswith(
            "decoder_query_counts"
        ):
            # Counts should not be incremented during batching
            return torch.tensor(
                [0] * value.dim(), dtype=torch.long
            )  # Or return 0 for scalar? Check PyG docs if needed. Assuming tensor counts.
        else:
            return super().__inc__(key, value, *args, **kwargs)


class UnifiedCollateFunction:
    """
    Unified collate function that handles online graph building.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        magno_radius: float = 0.033,
        magno_scales: List[float] = [1.0],
        latent_tokens: torch.Tensor = None,
        neighbor_search_method: str = "bidirectional",
        k_neighbors: int = 1,
        asynchronous_graph_building: bool = True,
    ):
        """
        Initialize unified collate function.

        Args:
            coord_dim: Coordinate dimension (2 or 3)
            magno_radius: Base radius for graph neighbor search
            magno_scales: List of scale factors for multi-scale graphs
            latent_tokens: Latent tokens coordinates
            neighbor_search_method: Method for neighbor search (bidirectional, radius, knn)
            k_neighbors: Number of neighbors for neighbor search
            asynchronous_graph_building: Whether to build graphs online
        """
        self.coord_dim = coord_dim
        self.magno_radius = magno_radius
        self.magno_scales = magno_scales
        self.neighbor_search_method = neighbor_search_method
        self.k_neighbors = k_neighbors
        self.asynchronous_graph_building = asynchronous_graph_building

        self.latent_queries = latent_tokens

    def __call__(self, batch: List[VTKMeshDataset]) -> Batch:
        """Collate function that processes a batch of samples.

        Args:s
            batch: List of samples from dataset

        Returns:
            Batch of samples
        """
        data_list: List[EnrichedData] = []
        for data in batch:
            if data.__class__.__name__ != "EnrichedData":
                enriched = EnrichedData(pos=data.pos, x=data.x)
                # Copy over all existing attributes except ones we explicitly set
                for attr, value in data:
                    if attr not in ["pos", "x"]:
                        setattr(enriched, attr, value)
                        print(attr, value.pos.mean().item(), value.c.mean().item())
                data = enriched
            data_list.append(data)

        # Always rebuild if online_graph_building is True
        if self.asynchronous_graph_building:
            # Prepare strategies
            enc_strategy, dec_strategy = parse_neighbor_strategy(
                self.neighbor_search_method
            )
            latent_tokens = self.latent_queries
            if latent_tokens is None:
                raise ValueError(
                    "latent_tokens must be provided for online graph building in collate function."
                )
            num_latent_nodes = int(latent_tokens.shape[0])

            # Ensure CPU tensors for neighbor search in workers
            latent_tokens_cpu = latent_tokens.cpu()

            for data in data_list:
                # Ensure attribute for batching increments
                data.num_latent_nodes = num_latent_nodes

                phys_pos = data.pos.to(torch.float32)
                num_phys_nodes = int(phys_pos.shape[0])
                batch_idx_phys = torch.zeros(num_phys_nodes, dtype=torch.long)
                batch_idx_latent = torch.zeros(num_latent_nodes, dtype=torch.long)

                for scale_idx, scale in enumerate(self.magno_scales):
                    scaled_radius = float(self.magno_radius) * float(scale)

                    # Encoder edges: phys -> latent, edge_index [2, E] = [phys_idx, latent_idx]
                    enc_edge_index = get_neighbor_strategy(
                        neighbor_strategy=enc_strategy,
                        phys_pos=phys_pos,
                        batch_idx_phys=batch_idx_phys,
                        latent_tokens_pos=latent_tokens_cpu,
                        batch_idx_latent=batch_idx_latent,
                        radius=scaled_radius,
                        k_neighbors=int(self.k_neighbors),
                        is_decoder=False,
                    ).to(dtype=torch.long)
                    setattr(data, f"encoder_edge_index_s{scale_idx}", enc_edge_index)
                    if enc_edge_index.numel() > 0:
                        enc_counts = torch.bincount(
                            enc_edge_index[1], minlength=num_latent_nodes
                        ).to(dtype=torch.long)
                    else:
                        enc_counts = torch.zeros(num_latent_nodes, dtype=torch.long)
                    setattr(data, f"encoder_query_counts_s{scale_idx}", enc_counts)

                    # Decoder edges: latent -> phys, edge_index [2, E] = [latent_idx, phys_idx]
                    dec_edge_index = get_neighbor_strategy(
                        neighbor_strategy=dec_strategy,
                        phys_pos=phys_pos,
                        batch_idx_phys=batch_idx_phys,
                        latent_tokens_pos=latent_tokens_cpu,
                        batch_idx_latent=batch_idx_latent,
                        radius=scaled_radius,
                        k_neighbors=int(self.k_neighbors),
                        is_decoder=True,
                    ).to(dtype=torch.long)
                    setattr(data, f"decoder_edge_index_s{scale_idx}", dec_edge_index)
                    if dec_edge_index.numel() > 0:
                        dec_counts = torch.bincount(
                            dec_edge_index[1], minlength=num_phys_nodes
                        ).to(dtype=torch.long)
                    else:
                        dec_counts = torch.zeros(num_phys_nodes, dtype=torch.long)
                    setattr(data, f"decoder_query_counts_s{scale_idx}", dec_counts)

        return Batch.from_data_list(data_list)


def create_collate_function(
    coord_dim: int = 2,
    magno_radius: float = 0.033,
    magno_scales: List[float] = [1.0],
    latent_tokens: torch.Tensor = None,
    neighbor_search_method: str = "bidirectional",
    k_neighbors: int = 1,
    asynchronous_graph_building: bool = True,
    **kwargs,
) -> UnifiedCollateFunction:
    """Factory function to create a collate function with appropriate configuration.

    Args:
        coord_dim: Coordinate dimension
        magno_radius: Base graph radius
        magno_scales: Graph scale factors
        latent_tokens: Latent tokens coordinates
        neighbor_search_method: Method for neighbor search (bidirectional, radius, knn)
        k_neighbors: Number of neighbors for neighbor search
        asynchronous_graph_building: Whether to build graphs online
        **kwargs: Additional configuration parameters

    Returns:
        Configured UnifiedCollateFunction instance
    """
    return UnifiedCollateFunction(
        coord_dim=coord_dim,
        magno_radius=magno_radius,
        magno_scales=magno_scales,
        latent_tokens=latent_tokens,
        neighbor_search_method=neighbor_search_method,
        k_neighbors=k_neighbors,
        asynchronous_graph_building=asynchronous_graph_building,
        **kwargs,
    )


class VTKMeshDataset(pyg_data.Dataset):
    """
    PyTorch Geometric Dataset for loading preprocessed VTK mesh data.
    Assumes data is preprocessed and saved as individual .pt files containing Data objects.

    Args:
        root (str): Root directory where the dataset should be saved/found.
                    Contains raw (optional) and processed directories.
        order_file (str): Path to the order.txt file.
        transform (callable, optional): Data transformation function applied after loading.
        pre_transform (callable, optional): Data transformation function applied before saving processed data.
        pre_filter (callable, optional): Data filtering function applied before saving.
    """

    def __init__(
        self,
        root: str,
        order_file: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.order_file = order_file
        # Assuming processed files are stored in root/processed/
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load indices after processing ensures processed files are available
        self._load_split_indices()

    def _load_split_indices(self):
        with open(self.order_file, "r") as f:
            all_filenames = [line for line in f.read().splitlines()]

        # Generate indices based on dataset size and split
        total_samples = len(all_filenames)
        indices = np.arange(total_samples)

        self.split_filenames = [f"{all_filenames[i]}.pt" for i in indices]
        print(
            f"Loaded {len(self.split_filenames)} samples from file: {self.order_file}"
        )

    def len(self):
        return len(self.split_filenames)

    def get(self, idx):
        filepath = os.path.join(self.root, self.split_filenames[idx])
        try:
            data = torch.load(filepath, weights_only=False)
            # data = torch.load(filepath, weights_only=False)
            # with paddle.device("cpu"):
            #     data = paddle.load(filepath.replace(".pt", ".pd"))
            # data = Data(
            #     pos=torch.from_dlpack(data["pos"]),
            #     x=torch.from_dlpack(data["x"]),
            #     c=torch.from_dlpack(data["c"]),
            # )
            # Apply normalization here if stats are available and not done in preprocessing
            # Example:
            # if hasattr(self, 'mean') and hasattr(self, 'std'):
            #    data.x = (data.x - self.mean) / (self.std + EPSILON)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Processed file not found: {filepath}. Ensure preprocessing script was run."
            )
        except Exception as e:
            print(f"Error loading data for index {idx} (file: {filepath}): {e}")
            raise e
