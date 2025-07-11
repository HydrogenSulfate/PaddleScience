# import random
from typing import Dict

import hydra

# import numpy as np
import paddle

# import scipy.io as sio
from model import Decoder
from model import Encoder
from omegaconf import DictConfig

import ppsci
from ppsci.data.dataset.tmtdataset import BatchParser

# from ppsci.utils import save_load
from ppsci.utils.misc import logger

# from ppsci.utils import misc


# from paddle.nn import functional as F


dtype = paddle.get_default_dtype()


def train(cfg: DictConfig):
    # Initialize model
    encoder = Encoder(
        **cfg.MODEL.encoder,
    )
    decoder = Decoder(
        **cfg.MODEL.decoder,
    )

    class FAE(ppsci.arch.base.Arch):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec

        def forward(self, batch: Dict[str, paddle.Tensor]):
            coords, x = batch["coords"], batch["x"]
            logger.debug(f"coords.shape = {coords.shape}")
            logger.debug(f"x.shape = {x.shape}")
            # coords: [1, num_query_points, 2] # 随机给定 num_query_points 个查询点
            # x: [b, h, w, 1]: 随机给定 h x w分辨率的物理场
            # y: [b, num_query_points, 1]: num_query_points 个查询点对应的物理场的值
            z = encoder(x)  # [b, l, c]
            logger.debug(f"z.shape = {z.shape}")

            u_pred = decoder(z, coords)
            return {
                "u": u_pred,
            }

    fae = FAE(encoder, decoder)
    print(
        f"Model storage cost: {fae.num_params * 4 / 1024 / 1024:.2f} MB of parameters"
    )

    # init constraint
    train_dataloader_cfg = {
        "dataset": {
            "name": "TMTDataset",
            "input_keys": cfg.input_keys,
            "label_keys": cfg.label_keys,
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
        train(cfg)
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
