# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.loss import mtl


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"NavierStokes": ppsci.equation.NavierStokes(cfg.NU, cfg.RHO, 2, False)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((-0.05, -0.05), (0.05, 0.05))}

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
        },
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "auto_collation": False,
    }

    # set constraint
    bc_top = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 1, "v": 0},
        geom["rect"],
        {
            "dataset": {
                "name": "NamedArrayDataset",
            },
            "sampler": {
                "name": "BatchSampler",
                "drop_last": True,
                "shuffle": True,
            },
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
            "auto_collation": False,
            "batch_size": cfg.TRAIN.batch_size.bc_top,
        },
        ppsci.loss.MSELoss("sum"),
        weight_dict={"u": lambda in_: 1 - 20 * in_["x"]},
        criteria=lambda x, y: np.isclose(y, 0.05),
        name="BC_top",
    )
    bc_noslip = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_noslip},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: (y < 0.05),
        name="BC_noslip",
    )
    pde = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.pde},
        ppsci.loss.MSELoss("sum"),
        weight_dict={
            "continuity": "sdf",
            "momentum_x": "sdf",
            "momentum_y": "sdf",
        },
        name="EQ",
    )
    # wrap constraints together
    constraint = {
        bc_top.name: bc_top,
        bc_noslip.name: bc_noslip,
        pde.name: pde,
    }

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # set validator
    data = ppsci.utils.reader.load_csv_file(
        "/workspace/hesensen/modulus_pd_th_bkd_compare/modulus-sym/examples/ldc/openfoam/cavity_uniformVel0.csv",
        ("x", "y", "u", "v", "p"),
        {"x": "Points:0", "y": "Points:1", "u": "U:0", "v": "U:1"},
    )
    data["x"] -= 0.05
    data["y"] -= 0.05
    eval_input = {"x": data["x"], "y": data["y"]}
    eval_label = {"u": data["u"], "v": data["v"]}
    valid_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": eval_input,
            "label": eval_label,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }
    uv_validator = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="openfoam_validator",
    )
    validator = {uv_validator.name: uv_validator}

    # set visualizer(optional)
    # manually collate input data for visualization,
    visualizer = {
        "visualize_u_v_p": ppsci.visualize.VisualizerVtu(
            eval_input,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            prefix="result_u_v_p",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        log_freq=cfg.log_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
        loss_aggregator=mtl.Relobralo(len(constraint)),
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"NavierStokes": ppsci.equation.NavierStokes(cfg.NU, cfg.RHO, 2, False)}

    # set validator
    # set validator
    data = ppsci.utils.reader.load_csv_file(
        "/workspace/hesensen/modulus_pd_th_bkd_compare/modulus-sym/examples/ldc/openfoam/cavity_uniformVel0.csv",
        ("x", "y", "u", "v", "p"),
        {"x": "Points:0", "y": "Points:1", "u": "U:0", "v": "U:1"},
    )
    data["x"] -= 0.05
    data["y"] -= 0.05
    eval_input = {"x": data["x"], "y": data["y"]}
    eval_label = {"u": data["u"], "v": data["v"]}
    valid_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": eval_input,
            "label": eval_label,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }
    uv_validator = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="openfoam_validator",
    )
    validator = {uv_validator.name: uv_validator}

    # set visualizer(optional)
    # manually collate input data for visualization,
    visualizer = {
        "visualize_u_v_p": ppsci.visualize.VisualizerVtu(
            eval_input,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            prefix="result_u_v_p",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()


@hydra.main(version_base=None, config_path="./conf", config_name="ldc2d_modulus.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
