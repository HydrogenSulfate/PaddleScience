# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from os import path as osp
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from model import FAE
from model import Decoder
from model import DiT
from model import Encoder
from omegaconf import DictConfig
from tqdm import tqdm

import ppsci
from ppsci.data.dataset.tmtdataset import BatchParser
from ppsci.utils import save_load
from ppsci.utils.misc import logger

dtype = paddle.get_default_dtype()


def train_fae(cfg: DictConfig):
    # Initialize model
    encoder = Encoder(
        **cfg.MODEL.encoder,
    )
    decoder = Decoder(
        **cfg.MODEL.decoder,
    )

    fae = FAE(
        cfg.MODEL.input_keys,
        cfg.MODEL.output_keys,
        encoder,
        decoder,
    )

    # init constraint
    train_dataloader_cfg = {
        "dataset": {
            "name": "TMTDataset",
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "data_path": cfg.DATA_PATH,
            "num_train": cfg.num_train,
            "mode": "train",
            "stage": cfg.stage,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 0,
    }

    sup_cst = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.MSELoss(),
    )
    # reset epochs & iters_per_epoch
    cfg.TRAIN.iters_per_epoch = len(sup_cst.data_loader)
    logger.debug(f"cfg.TRAIN.iters_per_epoch = {cfg.TRAIN.iters_per_epoch}")
    cfg.TRAIN.epochs = cfg.TRAIN.steps // len(sup_cst.data_loader)
    cfg.TRAIN.lr_scheduler.warmup_epoch /= len(sup_cst.data_loader)
    logger.debug(
        f"cfg.TRAIN.lr_scheduler.warmup_epoch = {cfg.TRAIN.lr_scheduler.warmup_epoch}"
    )

    # Create learning rate schedule and optimizer
    lr = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=len(sup_cst.data_loader),
        **cfg.TRAIN.lr_scheduler,
    )()
    optimizer = ppsci.optimizer.AdamW(
        lr,
        beta1=cfg.TRAIN.beta1,
        beta2=cfg.TRAIN.beta2,
        epsilon=cfg.TRAIN.eps,
        weight_decay=cfg.TRAIN.weight_decay,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(cfg.TRAIN.clip_norm),
    )(fae)

    # wrap batch parse to constraint
    sample_batch = next(iter(sup_cst.data_loader))
    b, h, w, c = sample_batch.shape

    class PostProcessDataLoader:
        def __init__(self, dataloader, parser: BatchParser):
            self.dataloader = dataloader
            self.parser = parser

        def __iter__(self):
            for batch in self.dataloader:
                yield self.parser.random_query(batch)

        def __len__(self):
            return len(self.dataloader)

    sup_cst.data_loader = PostProcessDataLoader(
        sup_cst.data_loader,
        BatchParser(cfg.TRAIN.num_queries, h, w, cfg.TRAIN.solution),
    )
    sup_cst.data_iter = iter(sup_cst.data_loader)

    # init solver
    solver = ppsci.solver.Solver(
        fae,
        {"sup": sup_cst},
        optimizer=optimizer,
        cfg=cfg,
    )
    # train
    solver.train()


def train_diffusion(cfg: DictConfig):
    # Initialize fae model
    encoder = Encoder(
        **cfg.FAE.encoder,
    )
    decoder = Decoder(
        **cfg.FAE.decoder,
    )
    fae = FAE(
        cfg.FAE.input_keys,
        cfg.FAE.output_keys,
        encoder,
        decoder,
    )
    save_load.load_pretrain(
        fae,
        cfg.FAE.pretrained_model_path,
    )
    fae.freeze()

    # Initialize diffusion model
    dit = DiT(**cfg.DIT)

    class ModelWrapper(paddle.nn.Layer):
        def __init__(self, enc: Encoder, dit: DiT):
            super().__init__()
            self.enc = enc
            self.dit = dit

        def forward(self, batch: Dict[str, paddle.Tensor]):
            with paddle.no_grad():
                u = batch["u"]
                v = batch["v"]
                z_u = self.enc(u)
                z_v = self.enc(v)
                z_c = paddle.concat([z_u, z_v], axis=-1)

                if self.training:
                    p = batch["p"]
                    sdf = batch["sdf"]
                    z_p = self.enc(p)
                    z_sdf = self.enc(sdf)
                    z_1 = paddle.concat([z_p, z_sdf], axis=-1)
                    z_0 = paddle.randn(z_1.shape)  # (b, 200, 512) 初始分布，随机采样
                    t = paddle.uniform(
                        [z_1.shape[0], *[1 for _ in range(z_1.ndim - 1)]]
                    )
                    z_t = t * (z_1 - z_0)
                    v_t = z_1 - z_0
                else:
                    t = batch["t"]
                    z_t = batch["z_t"]

            # only training dit
            v_t_pred = self.dit(z_t, t.flatten(), z_c)

            if self.training:
                return {
                    "v_t_err": v_t - v_t_pred,
                }
            else:
                return {
                    "v_t": v_t_pred,
                }

    model = ModelWrapper(
        encoder,
        decoder,
        dit,
    )

    # init constraint
    train_dataloader_cfg = {
        "dataset": {
            "name": "TMTDataset",
            "input_keys": cfg.FAE.input_keys,
            "label_keys": cfg.FAE.output_keys,
            "data_path": cfg.DATA_PATH,
            "num_train": cfg.num_train,
            "mode": "train",
            "stage": cfg.stage,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 0,
    }

    sup_cst = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(
            lambda i, l, w: {"v_t": (i["v_t_err"] ** 2).mean()}
        ),
    )

    # reset epochs & iters_per_epoch
    cfg.TRAIN.iters_per_epoch = len(sup_cst.data_loader)
    logger.debug(f"cfg.TRAIN.iters_per_epoch = {cfg.TRAIN.iters_per_epoch}")
    cfg.TRAIN.epochs = cfg.TRAIN.steps // len(sup_cst.data_loader)
    cfg.TRAIN.lr_scheduler.warmup_epoch /= len(sup_cst.data_loader)
    logger.debug(
        f"cfg.TRAIN.lr_scheduler.warmup_epoch = {cfg.TRAIN.lr_scheduler.warmup_epoch}"
    )

    # Create learning rate schedule and optimizer
    lr = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=len(sup_cst.data_loader),
        **cfg.TRAIN.lr_scheduler,
    )()
    optimizer = ppsci.optimizer.AdamW(
        lr,
        beta1=cfg.TRAIN.beta1,
        beta2=cfg.TRAIN.beta2,
        epsilon=cfg.TRAIN.eps,
        weight_decay=cfg.TRAIN.weight_decay,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(cfg.TRAIN.clip_norm),
    )(dit)

    # wrap batch parse to constraint
    sample_batch = next(iter(sup_cst.data_loader))
    b, h, w, c = sample_batch.shape

    class PostProcessDataLoader:
        def __init__(self, dataloader, parser: BatchParser):
            self.dataloader = dataloader
            self.parser = parser

        def __iter__(self):
            for batch in self.dataloader:
                yield self.parser.random_downsample(batch)

        def __len__(self):
            return len(self.dataloader)

    sup_cst.data_loader = PostProcessDataLoader(
        sup_cst.data_loader,
        BatchParser(None, h, w, cfg.TRAIN.solution),
    )
    sup_cst.data_iter = iter(sup_cst.data_loader)

    # init solver
    solver = ppsci.solver.Solver(
        model,
        {"sup": sup_cst},
        optimizer=optimizer,
        cfg=cfg,
    )
    # train
    solver.train()


@paddle.no_grad()
def evaluate(cfg: DictConfig):
    # Initialize encoder & decoder model
    encoder = Encoder(
        **cfg.FAE.encoder,
    )
    decoder = Decoder(
        **cfg.FAE.decoder,
    )
    dit = DiT(**cfg.DIT)

    class ModelWrapper(paddle.nn.Layer):
        def __init__(self, enc: Encoder, dec: Decoder, dit: DiT):
            super().__init__()
            self.enc = enc
            self.dec = dec
            self.dit = dit

        def forward(self, batch: Dict[str, paddle.Tensor]):
            u = batch["u"]
            v = batch["v"]
            z_u = self.enc(u)
            z_v = self.enc(v)
            z_c = paddle.concat([z_u, z_v], axis=-1)

            if self.training:
                p = batch["p"]
                sdf = batch["sdf"]
                z_p = self.enc(p)
                z_sdf = self.enc(sdf)
                z_1 = paddle.concat([z_p, z_sdf], axis=-1)
                z_0 = paddle.randn(z_1.shape)  # (b, 200, 512) 初始分布，随机采样
                t = paddle.uniform([z_1.shape[0], *[1 for _ in range(z_1.ndim - 1)]])
                z_t = t * (z_1 - z_0)
                v_t = z_1 - z_0
            else:
                t = batch["t"]
                z_t = batch["z_t"]

            # only training dit
            v_t_pred = self.dit(z_t, t.flatten(), z_c)

            if self.training:
                return {
                    "v_t_err": v_t - v_t_pred,
                }
            else:
                return {
                    "v_t": v_t_pred,
                }

    model = ModelWrapper(
        encoder,
        decoder,
        dit,
    )
    save_load.load_pretrain(model, cfg.EVAL.pretrained_model_path)

    # init evaluate data
    eval_dataset = ppsci.data.dataset.TMTDataset(
        input_keys=cfg.FAE.input_keys,
        label_keys=cfg.FAE.output_keys,
        data_path=cfg.DATA_PATH,
        num_train=cfg.num_train,
        mode="test",
        stage=cfg.stage,
    )
    eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=cfg.EVAL.batch_size)

    h, w = 200, 100
    x_coords = np.linspace(0, 1, h)
    y_coords = np.linspace(0, 1, w)
    x_coords, y_coords = np.meshgrid(x_coords, y_coords, indexing="ij")
    coords = np.hstack([x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)])[None, ...]

    # noise_level = 1.0
    d = 2
    u_input_list = []
    v_input_list = []
    p_pred_list = []
    sdf_pred_list = []
    p_true_list = []
    sdf_true_list = []

    def sample_ode(
        z0: paddle.Tensor = None,
        c: paddle.Tensor = None,
        num_steps: int = None,
        use_conditioning: bool = False,
    ):
        dt = 1 / num_steps
        traj = [z0]

        z = z0
        for i in tqdm(range(num_steps)):
            t = paddle.ones([z.shape[0]]) * i / num_steps
            if use_conditioning:
                pred = dit(z, t, c)
            else:
                pred = dit(z, t)
            z = z + pred * dt
            traj.append(z)
        return z, traj

    iters = 0
    for batch in tqdm(eval_loader):
        iters = iters + 1
        u: paddle.Tensor = batch[:, ::d, ::d, 0:1]
        v: paddle.Tensor = batch[:, ::d, ::d, 1:2]
        p: paddle.Tensor = batch[..., 2:3]
        sdf: paddle.Tensor = batch[..., 3:4]

        logger.debug(f"u.shape = {u.shape}")
        logger.debug(f"v.shape = {v.shape}")
        logger.debug(f"p.shape = {p.shape}")
        logger.debug(f"sdf.shape = {sdf.shape}")

        u = u  # + noise_level * paddle.randn(u.shape)
        v = v  # + noise_level * paddle.randn(v.shape)

        z_u = encoder(u)
        logger.debug(f"z_u.shape = {z_u.shape}")
        z_v = encoder(v)
        logger.debug(f"z_v.shape = {z_v.shape}")

        # z_p = encoder(p)
        # z_sdf = encoder(sdf)

        z_c = paddle.concat([z_u, z_v], axis=-1)  # (b, l, 2c)
        logger.debug(f"z_c.shape = {z_c.shape}")

        z0 = paddle.randn(shape=z_c.shape)
        z1_new, _ = sample_ode(
            z0=z0,
            c=z_c,
            num_steps=cfg.EVAL.num_steps,
            use_conditioning=cfg.EVAL.use_conditioning,
        )
        logger.debug(f"z1_new.shape = {z1_new.shape}")

        c_dim = z_c.shape[-1]
        z_p_new = z1_new[..., : c_dim // 2]
        z_sdf_new = z1_new[..., c_dim // 2 :]

        logger.debug(f"z_p_new.shape = {z_p_new.shape}")
        logger.debug(f"z_sdf_new.shape = {z_sdf_new.shape}")
        logger.debug(f"coords.shape = {coords.shape}")
        p_pred = decoder(z_p_new, coords)
        sdf_pred = decoder(z_sdf_new, coords)

        p_pred = p_pred.reshape([-1, h, w])
        sdf_pred = sdf_pred.reshape([-1, h, w])

        u_input_list.append(u)
        v_input_list.append(v)

        p_pred_list.append(p_pred)
        sdf_pred_list.append(sdf_pred)

        p_true_list.append(p)
        sdf_true_list.append(sdf)

        if iters == 4:
            break

    # Concatenate all results
    u_input = paddle.concat(u_input_list, axis=0).squeeze()
    # v_input = paddle.concat(v_input_list, axis=0).squeeze()
    p_pred = paddle.concat(p_pred_list, axis=0)
    sdf_pred = paddle.concat(sdf_pred_list, axis=0)
    p_true = paddle.concat(p_true_list, axis=0).squeeze()
    sdf_true = paddle.concat(sdf_true_list, axis=0).squeeze()

    def compute_error(pred, y):
        return paddle.linalg.norm(pred.flatten() - y.flatten()) / paddle.linalg.norm(
            y.flatten()
        )

    error = compute_error(p_pred, p_true)

    print("Mean relative error:", paddle.mean(error))
    print("Max relative error:", paddle.max(error))
    print("Min relative error:", paddle.min(error))
    print("Std relative error:", paddle.std(error))

    # Visualization of some examples
    k = 0
    # fig = plt.figure(figsize=(17, 4))
    plt.subplot(1, 4, 1)
    plt.title("Input")
    plt.imshow(u_input[k, :, :].T, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("Reference")
    plt.imshow(p_true[k, :, :].T, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(p_pred[k, :, :].T, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("Absolute Error")
    plt.imshow(paddle.abs(p_pred[k, :, :].T - p_true[k, :, :].T), cmap="jet")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(osp.join(cfg.output_dir, "Pressure predition of sample 1~4"))
    plt.close()

    k = 0
    # fig = plt.figure(figsize=(17, 4))
    plt.subplot(1, 4, 1)
    plt.title("Input")
    plt.imshow(u_input[k, :, :].T, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("Reference")
    plt.imshow(sdf_true[k, :, :].T, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(sdf_pred[k, :, :].T, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("Absolute Error")
    plt.imshow(paddle.abs(sdf_pred[k, :, :].T - sdf_true[k, :, :].T), cmap="jet")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(osp.join(cfg.output_dir, "SDF predition of sample 1~4"))
    plt.close()


# def export(cfg: DictConfig):
#     # set model
#     model = ppsci.arch.PirateNet(**cfg.MODEL)

#     # initialize solver
#     solver = ppsci.solver.Solver(model, cfg=cfg)
#     # export model
#     from paddle.static import InputSpec

#     input_spec = [
#         {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
#     ]
#     solver.export(input_spec, cfg.INFER.export_path, with_onnx=False)


# def inference(cfg: DictConfig):
#     from deploy.python_infer import pinn_predictor

#     predictor = pinn_predictor.PINNPredictor(cfg)
#     data = sio.loadmat(cfg.DATA_PATH)
#     u_ref = data["usol"].astype(dtype)  # (nt, nx)
#     t_star = data["t"].flatten().astype(dtype)  # [nt, ]
#     x_star = data["x"].flatten().astype(dtype)  # [nx, ]
#     tx_star = misc.cartesian_product(t_star, x_star).astype(dtype)

#     input_dict = {"t": tx_star[:, 0:1], "x": tx_star[:, 1:2]}
#     output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)
#     # mapping data to cfg.INFER.output_keys
#     output_dict = {
#         store_key: output_dict[infer_key]
#         for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
#     }
#     u_pred = output_dict["u"].reshape([len(t_star), len(x_star)])

# plot(t_star, x_star, u_ref, u_pred, cfg.output_dir)


@hydra.main(version_base=None, config_path="./conf", config_name="gifm_fae.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        if cfg.stage == "fae":
            train_fae(cfg)
        elif cfg.stage == "dit":
            train_diffusion(cfg)
        else:
            raise ValueError(f"cfg.stage should be 'fea', or 'dit, but got {cfg.stage}")
    elif cfg.mode == "eval":
        evaluate(cfg)
    # elif cfg.mode == "export":
    #     export(cfg)
    # elif cfg.mode == "infer":
    #     inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
