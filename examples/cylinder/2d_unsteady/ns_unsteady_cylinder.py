"""
Reference: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/ns_unsteady_cylinder
"""

from __future__ import annotations

import copy
from os import path as osp

import hydra
import matplotlib.tri as tri
import numpy as np
import paddle
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import jacobian as jac
from ppsci.loss import mtl
from ppsci.utils import logger

dtype = paddle.get_default_dtype()


def parabolic_inflow(y, U_max):
    u = 4 * U_max * y * (0.41 - y) / (0.41**2)
    return u


def get_dataset():
    data = np.load("data/ns_unsteady.npy", allow_pickle=True).item()
    u_ref = np.asarray(data["u"], dtype=dtype)
    v_ref = np.asarray(data["v"], dtype=dtype)
    p_ref = np.asarray(data["p"], dtype=dtype)
    coords = np.asarray(data["coords"], dtype=dtype)
    inflow_coords = np.asarray(data["inflow_coords"], dtype=dtype)
    outflow_coords = np.asarray(data["outflow_coords"], dtype=dtype)
    wall_coords = np.asarray(data["wall_coords"], dtype=dtype)
    cylinder_coords = np.asarray(data["cylinder_coords"], dtype=dtype)
    nu = np.asarray(data["nu"], dtype=dtype)

    return (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,
    )


def get_fine_mesh():
    data = np.load("./data/fine_mesh.npy", allow_pickle=True).item()
    fine_coords = np.asarray(data["coords"], dtype=dtype)

    data = np.load("./data/fine_mesh_near_cylinder.npy", allow_pickle=True).item()
    fine_coords_near_cyl = np.asarray(data["coords"], dtype=dtype)

    return fine_coords, fine_coords_near_cyl


def plot(t, x, y, u, v, p, w, triang, save_path):
    # Mask the triangles inside the cylinder
    center = (0.2, 0.2)
    radius = 0.05

    x_tri = x[triang.triangles].mean(axis=1)
    y_tri = y[triang.triangles].mean(axis=1)
    dist_from_center = np.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
    triang.set_mask(dist_from_center < radius)

    # Plot
    plt.rcParams.update(
        {
            "text.usetex": True,  # NOTE: This may cause error when using latex
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 20,
        }
    )
    fig1 = plt.figure(figsize=(18, 12))
    plt.suptitle(f"t = {t:.2f}")
    plt.subplot(4, 1, 1)
    plt.tricontourf(triang, u[-1], cmap="jet", levels=100)
    plt.colorbar()
    plt.title("Predicted $u$")
    plt.tight_layout()

    plt.subplot(4, 1, 2)
    plt.tricontourf(triang, v[-1], cmap="jet", levels=100)
    plt.colorbar()
    plt.title("Predicted $v$")
    plt.tight_layout()

    plt.subplot(4, 1, 3)
    plt.tricontourf(triang, p[-1], cmap="jet", levels=100)
    plt.colorbar()
    plt.title("Predicted $p$")
    plt.tight_layout()
    plt.show()

    plt.subplot(4, 1, 4)
    plt.tricontourf(triang, w[-1], cmap="jet", levels=100)
    plt.colorbar()
    plt.title("Predicted $w$")
    plt.tight_layout()
    plt.show()

    logger.message(f"Saving figure to: {save_path}")
    fig1.savefig(save_path, bbox_inches="tight", dpi=300)


def train(cfg: DictConfig):
    # Get dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cyl_coords,
        nu,
    ) = get_dataset()

    (
        fine_coords,
        _,
    ) = get_fine_mesh()  # finer mesh for evaluating PDE residuals

    noslip_coords = np.vstack((wall_coords, cyl_coords))

    # T_star = L_star / U_star  # characteristic time
    Re = cfg.U_star * cfg.L_star / nu

    # Nondimensionalize coordinates and inflow velocity
    # T = T / T_star
    inflow_coords = inflow_coords / cfg.L_star
    outflow_coords = outflow_coords / cfg.L_star
    noslip_coords = noslip_coords / cfg.L_star

    coords = coords / cfg.L_star
    fine_coords = fine_coords / cfg.L_star
    # fine_coords_near_cyl = fine_coords_near_cyl / cfg.L_star

    # Nondimensionalize flow field
    # u_inflow = u_inflow / cfg.U_star
    u_ref = u_ref / cfg.U_star
    v_ref = v_ref / cfg.U_star
    p_ref = p_ref / cfg.U_star**2

    # Temporal domain of each time window
    t0 = 0.0
    t1 = 1.0

    temporal_dom = np.asarray([t0, t1 * (1 + 0.05)], dtype=dtype)

    # Inflow boundary conditions
    U_max = 1.5  # maximum velocity

    def inflow_fn(y):
        return parabolic_inflow(y * cfg.L_star, U_max)

    # Set initial condition
    # Use the last time step of a coarse numerical solution as the initial condition
    u0 = u_ref[-1, :]
    v0 = v_ref[-1, :]
    p0 = p_ref[-1, :]

    for window_idx in range(cfg.NUM_TIME_WINDOWS):
        cfg_t = copy.deepcopy(cfg)
        cfg_t.output_dir = osp.join(cfg_t.output_dir, f"window_{window_idx}")
        logger.info(f"Training time window {window_idx + 1}")

        # set equation
        equation = {
            "NavierStokes": ppsci.equation.NavierStokes(1 / Re, 1.0, dim=2, time=True),
        }
        # set model
        model = ppsci.arch.ModifiedMLP(**cfg_t.MODEL)
        L, W = coords.max(axis=0) - coords.min(axis=0)

        model.register_input_transform(
            lambda input_dict: {
                "t": input_dict["t"] / temporal_dom[1],
                "x": input_dict["x"] / L,
                "y": input_dict["y"] / W,
            }
        )

        def output_transform(input_dict, output_dict):
            y_hat = input_dict["y"] * cfg_t.L_star
            return {
                "u": output_dict["u"] + 4 * 1.5 * y_hat * (0.41 - y_hat) / (0.41**2),
                "v": output_dict["v"],
                "p": output_dict["p"],
            }

        model.register_output_transform(output_transform)

        # set optimizer
        lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
            **cfg_t.TRAIN.lr_scheduler
        )()
        optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

        # set constraints
        # PDE constraint
        def res_gen_input_batch():
            t = np.random.uniform(
                temporal_dom[0],
                temporal_dom[1],
                size=(2 * cfg_t.TRAIN.batch_size.res, 1),
            ).astype(dtype)

            coarse_idx = np.random.choice(
                len(fine_coords),
                size=cfg_t.TRAIN.batch_size.res,
                replace=True,
            )

            fine_xy_half = fine_coords[coarse_idx]

            fine_idx = np.random.choice(
                len(fine_coords),
                size=cfg_t.TRAIN.batch_size.res,
                replace=True,
            )

            fine_xy = np.vstack([fine_xy_half, fine_coords[fine_idx]])
            fine_xy = np.random.permutation(fine_xy)
            return {
                "t": np.sort(t, axis=0),
                "x": fine_xy[:, 0:1],
                "y": fine_xy[:, 1:2],
            }

        def res_gen_label_batch(input_batch):
            return {
                "continuity": 0.0,
                "momentum_x": 0.0,
                "momentum_y": 0.0,
            }

        pde_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "ContinuousNamedArrayDataset",
                    "input": res_gen_input_batch,
                    "label": res_gen_label_batch,
                },
            },
            output_expr=equation["NavierStokes"].equations,
            loss=ppsci.loss.CausalMSELoss(
                cfg_t.TRAIN.causal.n_chunks,
                tol=cfg_t.TRAIN.causal.tol,
            ),
            name="PDE",
        )
        # INLET constraint
        def inflow_gen_input_batch():
            t = np.random.uniform(
                temporal_dom[0],
                temporal_dom[1],
                size=(cfg_t.TRAIN.batch_size.inflow, 1),
            ).astype(dtype)
            ind = np.random.choice(
                len(inflow_coords),
                size=cfg_t.TRAIN.batch_size.inflow,
            )
            xy = inflow_coords[ind]
            return {
                "t": t,
                "x": xy[:, 0:1],
                "y": xy[:, 1:2],
            }

        def inflow_gen_label_batch(input_batch):
            return {
                "u_inflow": inflow_fn(input_batch["y"]),
                "v_inflow": 0.0,
            }

        inflow_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "ContinuousNamedArrayDataset",
                    "input": inflow_gen_input_batch,
                    "label": inflow_gen_label_batch,
                },
            },
            output_expr={
                "u_inflow": lambda out: out["u"],
                "v_inflow": lambda out: out["v"],
            },
            loss=ppsci.loss.MSELoss("mean"),
            name="INLET",
        )
        # OUTLET constraint
        def outflow_gen_input_batch():
            t = np.random.uniform(
                temporal_dom[0],
                temporal_dom[1],
                size=(cfg_t.TRAIN.batch_size.outflow, 1),
            ).astype(dtype)
            ind = np.random.choice(
                len(outflow_coords),
                size=cfg_t.TRAIN.batch_size.outflow,
            )
            xy = outflow_coords[ind]
            return {
                "t": t,
                "x": xy[:, 0:1],
                "y": xy[:, 1:2],
            }

        def outflow_gen_label_batch(input_batch):
            return {
                "u_outflow": 0.0,
                "v_outflow": 0.0,
            }

        outflow_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "ContinuousNamedArrayDataset",
                    "input": outflow_gen_input_batch,
                    "label": outflow_gen_label_batch,
                },
            },
            output_expr={
                "u_outflow": lambda o: jac(o["u"], o["x"]) / Re - o["p"],
                "v_outflow": lambda o: jac(o["v"], o["x"]),
            },
            loss=ppsci.loss.MSELoss("mean"),
            name="OUTLET",
        )
        # NOSLIP constraint
        def noslip_gen_input_batch():
            t = np.random.uniform(
                temporal_dom[0],
                temporal_dom[1],
                size=(cfg_t.TRAIN.batch_size.noslip, 1),
            ).astype(dtype)
            ind = np.random.choice(
                len(noslip_coords),
                size=cfg_t.TRAIN.batch_size.noslip,
            )
            xy = noslip_coords[ind]
            return {
                "t": t,
                "x": xy[:, 0:1],
                "y": xy[:, 1:2],
            }

        def noslip_gen_label_batch(input_batch):
            return {
                "u_noslip": 0.0,
                "v_noslip": 0.0,
            }

        noslip_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "ContinuousNamedArrayDataset",
                    "input": noslip_gen_input_batch,
                    "label": noslip_gen_label_batch,
                },
            },
            output_expr={
                "u_noslip": lambda out: out["u"],
                "v_noslip": lambda out: out["v"],
            },
            loss=ppsci.loss.MSELoss("mean"),
            name="NOSLIP",
        )
        # IC constraint
        ic_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {
                        "t": np.zeros([len(coords), 1], dtype=dtype),
                        "x": coords[:, 0:1],
                        "y": coords[:, 1:2],
                    },
                    "label": {
                        "u_ic": u0[:, None],
                        "v_ic": v0[:, None],
                        "p_ic": p0[:, None],
                    },
                },
                "batch_size": cfg_t.TRAIN.batch_size.ic,
            },
            output_expr={
                "u_ic": lambda out: out["u"],
                "v_ic": lambda out: out["v"],
                "p_ic": lambda out: out["p"],
            },
            loss=ppsci.loss.MSELoss("mean"),
            name="IC",
        )
        # wrap constraints together
        constraint = {
            pde_constraint.name: pde_constraint,
            inflow_constraint.name: inflow_constraint,
            outflow_constraint.name: outflow_constraint,
            noslip_constraint.name: noslip_constraint,
            ic_constraint.name: ic_constraint,
        }

        # initialize solver
        solver = ppsci.solver.Solver(
            model,
            constraint,
            optimizer=optimizer,
            equation=equation,
            loss_aggregator=mtl.GradNorm(
                model,
                12,
                cfg_t.TRAIN.grad_norm.update_freq,
                cfg_t.TRAIN.grad_norm.momentum,
                list(cfg_t.TRAIN.grad_norm.init_weights),
            ),
            cfg=cfg_t,
        )
        # train model
        solver.train()

        # update initial condition for time marching algorithm
        if cfg_t.NUM_TIME_WINDOWS > 1:
            pred = solver.predict(
                {
                    "t": np.full([len(coords), 1], t1, dtype=dtype),
                    "x": coords[:, 0:1],
                    "y": coords[:, 1:2],
                },
                return_numpy=True,
            )
            u0, v0, p0 = pred["u"], pred["v"], pred["p"]
            u0 = u0.reshape(-1)
            v0 = v0.reshape(-1)
            p0 = p0.reshape(-1)
            logger.info(f"{u0.shape}, {v0.shape}, {p0.shape}")


def evaluate(cfg: DictConfig):
    # Load dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,
    ) = get_dataset()

    T = 1.0  # final time
    T_star = cfg.L_star / cfg.U_star  # characteristic time
    # T_star = cfg.L_star / cfg.U_star  # characteristic time
    # Re = cfg.U_star * cfg.L_star / nu

    # Nondimensionalize coordinates and inflow velocity
    T = T / T_star
    inflow_coords = inflow_coords / cfg.L_star
    outflow_coords = outflow_coords / cfg.L_star
    wall_coords = wall_coords / cfg.L_star
    cylinder_coords = cylinder_coords / cfg.L_star
    coords = coords / cfg.L_star

    # Nondimensionalize flow field
    # u_inflow = u_inflow / cfg.U_star
    u_ref = u_ref / cfg.U_star
    v_ref = v_ref / cfg.U_star
    p_ref = p_ref / cfg.U_star**2

    # Temporal domain of each time window
    t0 = 0.0
    t1 = 1.0

    temporal_dom = np.asarray(
        [t0, t1 * (1 + 0.05)], dtype=dtype
    )  # Must be same as the one used in training
    t_coords = np.linspace(0, t1, 20, dtype=dtype)[:-1]
    L, W = coords.max(axis=0) - coords.min(axis=0)

    # set model
    model = ppsci.arch.ModifiedMLP(**cfg.MODEL)
    model.register_input_transform(
        lambda input_dict: {
            "t": input_dict["t"] / temporal_dom[1],
            "x": input_dict["x"] / L,
            "y": input_dict["y"] / W,
        }
    )

    def output_transform(input_dict, output_dict):
        y_hat = input_dict["y"] * cfg.L_star
        return {
            "u": output_dict["u"] + 4 * 1.5 * y_hat * (0.41 - y_hat) / (0.41**2),
            "v": output_dict["v"],
            "p": output_dict["p"],
        }

    model.register_output_transform(output_transform)

    solver = ppsci.solver.Solver(
        model,
        cfg=cfg,
    )

    for idx in range(1):
        # window_1: 0~0.95
        # window_2: 1.0~1.95
        # ...
        # window_t: t-1~(t-0.05)
        ppsci.utils.load_pretrain(solver.model, cfg.EVAL.pretrained_model_path)

        def infer_one_window(ts):
            us, vs, ps, ws = [], [], [], []
            for t in ts:
                out = solver.predict(
                    {
                        "t": np.full([len(coords), 1], t, dtype=dtype),
                        "x": coords[:, 0:1],
                        "y": coords[:, 1:2],
                    },
                    batch_size=cfg.EVAL.batch_size,
                    expr_dict={
                        "u": lambda o: o["u"],
                        "v": lambda o: o["v"],
                        "p": lambda o: o["p"],
                        "w": lambda o: jac(o["v"], o["x"]) - jac(o["u"], o["y"]),
                    },
                    no_grad=False,
                    return_numpy=True,
                )
                u, v, p, w = out["u"], out["v"], out["p"], out["w"]
                us.append(u)
                vs.append(v)
                ps.append(p)
                ws.append(w)
            return (
                np.stack(us, 0)[..., 0],
                np.stack(vs, 0)[..., 0],
                np.stack(ps, 0)[..., 0],
                np.stack(ws, 0)[..., 0],
            )

        u_pred, v_pred, p_pred, w_pred = infer_one_window(t_coords)

    coords = coords * cfg.L_star

    u_ref = u_ref * cfg.U_star
    v_ref = v_ref * cfg.U_star

    u_pred = u_pred * cfg.U_star
    v_pred = v_pred * cfg.U_star

    x = coords[:, 0]
    y = coords[:, 1]
    triang = tri.Triangulation(x, y)

    plot(
        t_coords[-1],
        x,
        y,
        u_pred,
        v_pred,
        p_pred,
        w_pred,
        triang,
        "./ns_unsteady_pred.png",
    )


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.ModifiedMLP(**cfg.MODEL)

    # Load dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,
    ) = get_dataset()

    # Temporal domain of each time window
    t0 = 0.0
    t1 = 1.0
    temporal_dom = np.asarray(
        [t0, t1 * (1 + 0.05)], dtype=dtype
    )  # Must be same as the one used in training
    L, W = coords.max(axis=0) - coords.min(axis=0)

    model.register_input_transform(
        lambda input_dict: {
            "t": input_dict["t"] / temporal_dom[1],
            "x": input_dict["x"] / L,
            "y": input_dict["y"] / W,
        }
    )

    def output_transform(input_dict, output_dict):
        y_hat = input_dict["y"] * cfg.L_star
        return {
            "u": output_dict["u"] + 4 * 1.5 * y_hat * (0.41 - y_hat) / (0.41**2),
            "v": output_dict["v"],
            "p": output_dict["p"],
        }

    model.register_output_transform(output_transform)

    # initialize solver
    solver = ppsci.solver.Solver(model, cfg=cfg)
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]
    solver.export(input_spec, cfg.INFER.export_path, with_onnx=False)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)
    # Load dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,
    ) = get_dataset()

    # T_star = cfg.L_star / cfg.U_star  # characteristic time
    # Re = cfg.U_star * cfg.L_star / nu

    # Nondimensionalize coordinates and inflow velocity
    # T = T / T_star
    inflow_coords = inflow_coords / cfg.L_star
    outflow_coords = outflow_coords / cfg.L_star
    wall_coords = wall_coords / cfg.L_star
    cylinder_coords = cylinder_coords / cfg.L_star
    coords = coords / cfg.L_star

    # Nondimensionalize flow field
    # u_inflow = u_inflow / cfg.U_star
    u_ref = u_ref / cfg.U_star
    v_ref = v_ref / cfg.U_star
    p_ref = p_ref / cfg.U_star**2

    # Temporal domain of each time window
    t1 = 1.0
    t_coords = np.linspace(0, t1, 20, dtype=dtype)[:-1]

    for idx in range(1):

        def infer_time(ts):
            us, vs, ps = [], [], []
            for t in ts:
                out = predictor.predict(
                    {
                        "t": np.full([len(coords), 1], t, dtype=dtype),
                        "x": coords[:, 0:1],
                        "y": coords[:, 1:2],
                    },
                    batch_size=cfg.EVAL.batch_size,
                    return_numpy=True,
                )
                u, v, p = out["u"], out["v"], out["p"]
                us.append(u)
                vs.append(v)
                ps.append(p)
            return (
                np.stack(us, 0)[..., 0],
                np.stack(vs, 0)[..., 0],
                np.stack(ps, 0)[..., 0],
            )

        u, v, p = infer_time(t_coords)

    coords = coords * cfg.L_star

    u = u * cfg.U_star
    v = v * cfg.U_star

    x = coords[:, 0]
    y = coords[:, 1]
    triang = tri.Triangulation(x, y)

    plot(t_coords[-1], x, y, u, v, p, triang, "./ns_unsteady_pred.png")


@hydra.main(
    version_base=None, config_path="./conf", config_name="ns_unsteady_cylinder.yaml"
)
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
