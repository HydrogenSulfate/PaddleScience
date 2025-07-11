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

import hydra
import paddle
from model import FAE
from model import Decoder
from model import Encoder
from omegaconf import DictConfig

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
    # for k, v in fae.state_dict().items():
    #     print(f"{k}#{v.shape}#{v.mean().item():.10f}#{v.std():.10f}")
    # exit()
    # # paddle.save(fae.state_dict(), "fae_shape.pd")
    # # exit()
    # save_load.load_pretrain(
    #     fae,
    #     "/work/PaddleScience/examples/_gifm/fae.pdparams"
    # )
    # print(f"Number of model parameters: {fae.num_params}")
    # coords = paddle.to_tensor(np.load("/gnodiff/fundiff/turbulence_mass_transfer/coords.npy"))
    # inp = paddle.to_tensor(np.load("/gnodiff/fundiff/turbulence_mass_transfer/inp.npy"))
    # u = paddle.to_tensor(np.load("/gnodiff/fundiff/turbulence_mass_transfer/u.npy"))
    # # u_pred = paddle.to_tensor(np.load("/gnodiff/fundiff/turbulence_mass_transfer/u_pred.npy"))
    # z_pred = paddle.to_tensor(np.load("/gnodiff/fundiff/turbulence_mass_transfer/z_pred.npy"))

    # out = fae({"coord": coords, "x": inp})["u"]

    # print(out.shape)
    # print(out.min().item(), out.max().item(), out.mean().item())
    # # np.testing.assert_allclose(out.numpy(), u_pred.numpy())
    # np.testing.assert_allclose(out.numpy(), z_pred.numpy(), 1e-6, 1e-6)
    # exit()

    # init constraint
    train_dataloader_cfg = {
        "dataset": {
            "name": "TMTDataset",
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "data_path": cfg.DATA_PATH,
            "num_train": cfg.num_train,
            "mode": "train",
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
    save_load.load_pretrain(cfg.TRAIN.pretrained_model_path)
    print(f"Number of model parameters: {fae.num_params}")

    # init constraint
    train_dataloader_cfg = {
        "dataset": {
            "name": "TMTDataset",
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "data_path": cfg.DATA_PATH,
            "num_train": cfg.num_train,
            "mode": "train",
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 2,
    }

    sup_cst = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.MSELoss(),
    )
    # reset epochs & iters_per_epoch
    cfg.TRAIN.iters_per_epoch = len(sup_cst.data_loader)
    logger.debug(f"cfg.TRAIN.iters_per_epoch = {cfg.TRAIN.iters_per_epoch}")
    cfg.TRAIN.epochs = cfg.TRAIN.steps // len(sup_cst.data_loader)

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


def train_dit(cfg: DictConfig):
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
    save_load.load_pretrain(
        fae,
        cfg.TRAIN.pretrained_model_path,
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
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 2,
    }

    sup_cst = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.MSELoss(),
    )
    # reset epochs
    cfg.TRAIN.epochs = cfg.TRAIN.steps // len(sup_cst.data_loader)

    # Create learning rate schedule and optimizer
    lr = ppsci.optimizer.lr_scheduler.ExponentialDecay(
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


# def evaluate(cfg: DictConfig):
#     # set model
#     model = ppsci.arch.PirateNet(**cfg.MODEL)

#     data = sio.loadmat(cfg.DATA_PATH)
#     u_ref = data["usol"].astype(dtype)  # (nt, nx)
#     t_star = data["t"].flatten().astype(dtype)  # [nt, ]
#     x_star = data["x"].flatten().astype(dtype)  # [nx, ]

#     # set validator
#     tx_star = misc.cartesian_product(t_star, x_star).astype(dtype)
#     eval_data = {"t": tx_star[:, 0:1], "x": tx_star[:, 1:2]}
#     eval_label = {"u": u_ref.reshape([-1, 1])}
#     u_validator = ppsci.validate.SupervisedValidator(
#         {
#             "dataset": {
#                 "name": "NamedArrayDataset",
#                 "input": eval_data,
#                 "label": eval_label,
#             },
#             "batch_size": cfg.EVAL.batch_size,
#         },
#         ppsci.loss.MSELoss("mean"),
#         {"u": lambda out: out["u"]},
#         metric={"L2Rel": ppsci.metric.L2Rel()},
#         name="u_validator",
#     )
#     validator = {u_validator.name: u_validator}

#     # initialize solver
#     solver = ppsci.solver.Solver(
#         model,
#         validator=validator,
#         cfg=cfg,
#     )

#     # evaluate after finished training
#     solver.eval()
#     # visualize prediction after finished training
#     u_pred = solver.predict(
#         eval_data, batch_size=cfg.EVAL.batch_size, return_numpy=True
#     )["u"]
#     u_pred = u_pred.reshape([len(t_star), len(x_star)])

#     # plot
#     # plot(t_star, x_star, u_ref, u_pred, cfg.output_dir)


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
            train_dit(cfg)
        else:
            raise ValueError(f"cfg.stage should be 'fea', or 'dit, but got {cfg.stage}")
    # elif cfg.mode == "eval":
    #     evaluate(cfg)
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
