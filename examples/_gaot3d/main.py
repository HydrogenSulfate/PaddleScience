"""
Reference: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/allen_cahn
"""

from __future__ import annotations

import hydra
import numpy as np
import paddle
import torch
from dataset import NormalizeFeatures
from dataset import RescalePosition
from dataset import VTKMeshDataset
from dataset import create_collate_function
from model import GAOT3D
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

import ppsci
from ppsci.utils import logger

dtype = paddle.get_default_dtype()


def train(cfg: DictConfig):
    # model
    pass


@paddle.no_grad()
def evaluate(cfg: DictConfig):
    model = GAOT3D(
        ["x"],
        ["c"],
        cfg.MODEL.input_size,
        cfg.MODEL.output_size,
        cfg.MODEL.magno,
        cfg.MODEL.transformer,
        latent_tokens=cfg.MODEL.latent_tokens,
    )
    ppsci.utils.load_pretrain(
        model,
        "./pressure_from_hss_trained_pt",
    )
    stats = paddle.load("drivaernet_fullpressure_norm_stats.pd")
    u_mean = stats["mean"].to(dtype)
    u_std = stats["std"].to(dtype)
    logger.info(f"Loaded x - Mean: {u_mean.numpy()}, Std: {u_std.numpy()}")

    if "c_mean" in stats and "c_std" in stats:
        c_mean = stats["c_mean"].to(dtype)
        c_std = stats["c_std"].to(dtype)
        logger.info(f"Loaded c - Mean: {c_mean.numpy()}, Std: {c_std.numpy()}")
    else:
        c_mean = None
        c_std = None
    # # Move data to device
    # batch = batch.to(self.device)
    # latent_tokens_dev = self.latent_tokens.to(self.device)
    phy_domain = ([-1.16, -1.20, 0.0], [4.21, 1.19, 1.77])
    x_min, y_min, z_min = phy_domain[0]
    x_max, y_max, z_max = phy_domain[1]
    meshgrid = paddle.meshgrid(
        paddle.linspace(x_min, x_max, cfg.MODEL.latent_tokens[0]),
        paddle.linspace(y_min, y_max, cfg.MODEL.latent_tokens[1]),
        paddle.linspace(z_min, z_max, cfg.MODEL.latent_tokens[2]),
        indexing="ij",
    )
    latent_queries = paddle.stack(meshgrid, axis=-1).reshape(-1, 3)

    def rescale(x: paddle.Tensor, lims=(-1, 1)) -> paddle.Tensor:
        return (x - x.min()) / (x.max() - x.min()) * (lims[1] - lims[0]) + lims[0]

    latent_tokens = rescale(latent_queries, (-1, 1))

    # ppsci.data.register_to_dataset(VTKMeshDataset)
    rescale_transform = RescalePosition(lims=(-1.0, 1.0))
    normalize_transform = NormalizeFeatures(
        mean=torch.tensor(u_mean.numpy(), device="cpu"),
        std=torch.tensor(u_std.numpy(), device="cpu"),
        c_mean=torch.tensor(c_mean.numpy(), device="cpu"),
        c_std=torch.tensor(c_std.numpy(), device="cpu"),
    )
    composed_transform = Compose([rescale_transform, normalize_transform])
    dataset = VTKMeshDataset(
        root="dataset/drivaernet/processed_pyg_normals",
        order_file="./dataset/drivaernet/test.txt",
        transform=composed_transform,
    )
    collate_fn = create_collate_function(
        coord_dim=cfg.MODEL.magno.gno_coord_dim,
        magno_radius=cfg.MODEL.magno.gno_radius,
        magno_scales=cfg.MODEL.magno.scales,
        latent_tokens=cfg.MODEL.latent_tokens,
        neighbor_search_method=cfg.MODEL.magno.neighbor_strategy,
        k_neighbors=cfg.MODEL.magno.k_neighbors,
        asynchronous_graph_building=cfg.MODEL.magno.asynchronous_graph_building,
    )
    test_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=False,
    )
    # print(len(test_dataloader))
    MEAN = paddle.tensor(-93.4105, dtype=dtype)  # -94.5
    STD = paddle.tensor(120.7879, dtype=dtype)  # 117.25
    mse_list = []
    mae_list = []
    rmse_list = []
    max_error_list = []
    rel_l2_list = []
    rel_l1_list = []
    total_samples = 0
    model.eval()
    print(model.num_params)
    for i, batch in enumerate(tqdm(test_dataloader)):
        batch = batch.to("cuda")
        # Model inference
        pred_norm = model(batch, latent_tokens)

        # Count samples
        batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        total_samples += batch_size

        target_norm = paddle.from_dlpack(batch.x)

        pred_denorm = pred_norm * u_std + u_mean
        target_denorm = target_norm * u_std + u_mean

        gtr = target_denorm
        prd = pred_denorm
        gtr_norm = (gtr - MEAN) / STD
        prd_norm = (prd - MEAN) / STD

        gtr_norm = gtr_norm.cpu().numpy()
        prd_norm = prd_norm.cpu().numpy()

        mse = np.mean((gtr_norm - prd_norm) ** 2)
        mae = np.mean(np.abs(gtr_norm - prd_norm))
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(gtr_norm - prd_norm))
        rel_l2 = np.mean(
            np.linalg.norm(gtr_norm - prd_norm, axis=0)
            / np.linalg.norm(gtr_norm, axis=0)
        )
        rel_l1 = np.mean(
            np.sum(np.abs(gtr_norm - prd_norm), axis=0)
            / np.sum(np.abs(gtr_norm), axis=0)
        )
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        max_error_list.append(max_error)
        rel_l2_list.append(rel_l2)
        rel_l1_list.append(rel_l1)
        del batch

    print(f"mse (x10^-2) = {np.mean(mse_list) * 100:.4f}")
    print(f"mae (x10^-1) = {np.mean(mae_list) * 10:.4f}")
    print(f"rmse = {np.mean(rmse_list):.4f}")
    print(f"max_error = {np.mean(max_error_list):.4f}")
    print(f"Rel L2 Error (%) = {np.mean(rel_l2_list) * 100:.4f}")
    print(f"Rel L1 Error (%) = {np.mean(rel_l1_list) * 100:.4f}")


def export(cfg: DictConfig):
    pass


def inference(cfg: DictConfig):
    pass


@hydra.main(version_base=None, config_path="./conf", config_name="pressure.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
