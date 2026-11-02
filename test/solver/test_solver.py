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

import gc
import math
from contextlib import nullcontext

import numpy as np
import paddle
import pytest

import ppsci
from ppsci.utils import save_load


@pytest.fixture(autouse=True)
def _force_cpu_device():
    """Run every test on CPU and restore the previous device afterwards."""
    prev_device = paddle.device.get_device()
    paddle.device.set_device("cpu")
    paddle.seed(0)
    yield
    paddle.device.set_device(prev_device)


def _make_model():
    return ppsci.arch.MLP(("x", "y"), ("u",), 2, 8)


def _make_optimizer(model):
    return ppsci.optimizer.Adam(1e-3)(model)


def _make_interior_constraint(iters_per_epoch=1, batch_size=8):
    geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
    return ppsci.constraint.InteriorConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        geom,
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": iters_per_epoch,
            "batch_size": batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )


def _make_supervised_validator(num_samples=20, keep_batch=False):
    rng = np.random.default_rng(0)
    input_data = {
        "x": rng.random((num_samples, 1), dtype="float32"),
        "y": rng.random((num_samples, 1), dtype="float32"),
    }
    label_data = {"u": rng.random((num_samples, 1), dtype="float32")}
    dataset = ppsci.data.dataset.NamedArrayDataset(input_data, label_data)
    dataloader_cfg = {
        "dataset": dataset,
        "batch_size": 8,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    return ppsci.validate.SupervisedValidator(
        dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        {"u": lambda out: out["u"]},
        metric={"MSE": ppsci.metric.MSE(keep_batch=keep_batch)},
        name="sup",
    )


def test_solver_train_cpu_and_latest_checkpoint(tmp_path):
    model = _make_model()
    optimizer = _make_optimizer(model)
    constraint = _make_interior_constraint()
    solver = ppsci.solver.Solver(
        model,
        {"EQ": constraint},
        output_dir=str(tmp_path),
        optimizer=optimizer,
        epochs=1,
        iters_per_epoch=2,
        log_freq=100,
        device="cpu",
    )

    before = {
        name: param.detach().numpy().copy() for name, param in model.named_parameters()
    }

    solver.train()

    assert solver.global_step == 2
    assert solver.max_steps == 2

    params_changed = any(
        not np.allclose(before[name], param.detach().numpy())
        for name, param in model.named_parameters()
    )
    assert params_changed

    ckpt_dir = tmp_path / "checkpoints"
    for suffix in (".pdparams", ".pdopt", ".pdstates"):
        assert (ckpt_dir / f"latest{suffix}").is_file()

    states = paddle.load(str(ckpt_dir / "latest.pdstates"))
    assert states["epoch"] == 1
    # No validator is used during plain training, so the metric stays infinite.
    assert math.isinf(states["metric"])


def test_solver_rejects_grad_norm_with_to_static(tmp_path):
    model = _make_model()
    optimizer = _make_optimizer(model)
    constraint = _make_interior_constraint()

    with pytest.raises(
        NotImplementedError,
        match=r"mtl\.GradNorm is currently incompatible with to_static=True/PIR",
    ):
        ppsci.solver.Solver(
            model,
            {"EQ": constraint},
            output_dir=str(tmp_path),
            optimizer=optimizer,
            device="cpu",
            to_static=True,
            loss_aggregator=ppsci.loss.mtl.GradNorm(model, num_losses=1),
        )


@pytest.mark.parametrize("optimizer_name", ["Adam", "LBFGS"])
@pytest.mark.parametrize("world_size", [1, 2])
def test_single_batch_loader_restart_gc_policy(
    tmp_path, monkeypatch, optimizer_name, world_size
):
    """Skip full GC on one device, but preserve PR #1251's distributed order.

    Collectives are mocked: this tests the training-loop policy, not the
    original multi-GPU allocator failure.
    """
    model = _make_model()
    optimizer = getattr(ppsci.optimizer, optimizer_name)(1e-3)(model)
    constraint = _make_interior_constraint()
    solver = ppsci.solver.Solver(
        model,
        {"EQ": constraint},
        output_dir=str(tmp_path),
        optimizer=optimizer,
        epochs=1,
        iters_per_epoch=3,
        device="cpu",
    )
    from ppsci.solver import train

    events = []

    class Synchronized:
        def __init__(self, enabled):
            self.enabled = enabled

        def __enter__(self):
            if self.enabled:
                events.append("barrier")

        def __exit__(self, *args):
            if self.enabled:
                events.append("barrier")

    solver.world_size = world_size
    monkeypatch.setattr(train.misc, "Synchronized", Synchronized)
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(solver, "no_sync_context_manager", lambda *args: nullcontext())
    monkeypatch.setattr(train.hpu, "fused_allreduce_gradients", lambda *args: None)
    solver.max_steps = solver.iters_per_epoch
    train_fn = (
        train.train_LBFGS_epoch_func
        if optimizer_name == "LBFGS"
        else train.train_epoch_func
    )
    train_fn(solver, 1, 100)

    assert solver.global_step == 3
    assert events == (["barrier", "gc", "barrier"] * 2 if world_size > 1 else [])


def test_solver_predict_batch_modes_and_eval_mode_restore(tmp_path):
    model = _make_model()
    solver = ppsci.solver.Solver(model, output_dir=str(tmp_path), device="cpu")

    rng = np.random.default_rng(0)
    input_dict = {
        "x": rng.random((100, 1), dtype="float32"),
        "y": rng.random((100, 1), dtype="float32"),
    }

    model.train()
    out_full = solver.predict(input_dict, batch_size=None)
    assert model.training is True

    out_batched = solver.predict(input_dict, batch_size=64)
    assert model.training is True

    assert tuple(out_full["u"].shape) == (100, 1)
    assert tuple(out_batched["u"].shape) == (100, 1)
    np.testing.assert_allclose(
        out_full["u"].numpy(),
        out_batched["u"].numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    out_numpy = solver.predict(input_dict, batch_size=64, return_numpy=True)
    assert model.training is True
    assert isinstance(out_numpy["u"], np.ndarray)
    np.testing.assert_allclose(
        out_numpy["u"], out_batched["u"].numpy(), rtol=1e-6, atol=1e-6
    )


def test_solver_predict_with_expr_dict(tmp_path):
    model = _make_model()
    solver = ppsci.solver.Solver(model, output_dir=str(tmp_path), device="cpu")

    rng = np.random.default_rng(0)
    input_dict = {
        "x": rng.random((100, 1), dtype="float32"),
        "y": rng.random((100, 1), dtype="float32"),
    }

    output_dict = solver.predict(
        input_dict,
        expr_dict={"u2": lambda out: out["u"] * 2},
        batch_size=64,
    )

    assert set(output_dict) == {"u", "u2"}
    assert tuple(output_dict["u2"].shape) == (100, 1)
    np.testing.assert_allclose(
        output_dict["u2"].numpy(),
        output_dict["u"].numpy() * 2,
        rtol=1e-6,
        atol=1e-6,
    )


def test_solver_eval_with_supervised_validator(tmp_path):
    model = _make_model()
    validator = _make_supervised_validator()
    solver = ppsci.solver.Solver(
        model,
        validator={"sup": validator},
        output_dir=str(tmp_path),
        device="cpu",
    )

    model.train()
    cur_metric, metric_group = solver.eval()

    assert isinstance(cur_metric, float)
    assert math.isfinite(cur_metric)
    assert set(metric_group) == {"MSE"}
    assert set(metric_group["MSE"]) == {"u"}
    assert metric_group["MSE"]["u"] == pytest.approx(cur_metric, abs=0.0)
    # `eval` is decorated to restore the previous training state.
    assert model.training is True


def test_solver_rejects_metric_keep_batch_mismatch(tmp_path):
    model = _make_model()
    # Default `compute_metric_by_batch` is False, so `keep_batch=True` is invalid.
    validator = _make_supervised_validator(keep_batch=True)

    with pytest.raises(ValueError, match="keep_batch"):
        ppsci.solver.Solver(
            model,
            validator={"sup": validator},
            output_dir=str(tmp_path),
            device="cpu",
        )


def test_solver_resume_restores_best_metric_and_epoch(tmp_path):
    output_dir = str(tmp_path)

    # Create an initial checkpoint at epoch 7 with a finite best metric.
    source_model = _make_model()
    source_optimizer = _make_optimizer(source_model)
    save_load.save_checkpoint(
        source_model,
        source_optimizer,
        {"metric": 0.123, "epoch": 7},
        output_dir=output_dir,
        prefix="resume",
    )
    expected_state = {
        name: param.detach().numpy().copy()
        for name, param in source_model.named_parameters()
    }

    model = _make_model()
    optimizer = _make_optimizer(model)
    constraint = _make_interior_constraint(iters_per_epoch=1, batch_size=8)
    solver = ppsci.solver.Solver(
        model,
        {"EQ": constraint},
        output_dir=output_dir,
        optimizer=optimizer,
        epochs=8,
        iters_per_epoch=1,
        log_freq=100,
        device="cpu",
        checkpoint_path=str(tmp_path / "checkpoints" / "resume"),
    )

    assert solver.best_metric == {"metric": 0.123, "epoch": 7}
    for name, param in model.named_parameters():
        np.testing.assert_allclose(
            param.detach().numpy(), expected_state[name], rtol=1e-6, atol=1e-6
        )

    # Training resumes at epoch 8 and runs exactly one iteration.
    solver.train()
    assert solver.global_step == 8
    assert solver.max_steps == 8

    states = paddle.load(str(tmp_path / "checkpoints" / "latest.pdstates"))
    assert states["epoch"] == 8


def test_solver_export_restores_static_mode_on_failure(tmp_path, monkeypatch):
    model = _make_model()
    solver = ppsci.solver.Solver(model, device="cpu")

    calls = []
    monkeypatch.setattr(paddle.jit, "enable_to_static", lambda flag: calls.append(flag))
    monkeypatch.setattr(paddle.jit, "to_static", lambda *args, **kwargs: model)

    def _fail(*args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(paddle.jit, "save", _fail)

    input_spec = [
        {
            key: paddle.static.InputSpec([None, 1], "float32", name=key)
            for key in model.input_keys
        }
    ]
    with pytest.raises(RuntimeError, match="save failed"):
        solver.export(input_spec, str(tmp_path / "inference" / "model"))

    # Even if jit.save raises, enable_to_static(False) must still run.
    assert calls == [True, False]


def test_solver_to_static_with_symbolic_equation(tmp_path):
    core = paddle.framework.core
    if not (
        hasattr(core, "_set_prim_all_enabled") and hasattr(core, "_is_all_prim_enabled")
    ):
        pytest.skip("primitive op decomposition API is unavailable")

    prev_prim = core._is_all_prim_enabled()
    try:
        model = _make_model()
        optimizer = _make_optimizer(model)
        equation = ppsci.equation.Laplace(2)
        geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
        constraint = ppsci.constraint.InteriorConstraint(
            equation.equations,
            {"laplace": 0},
            geom,
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": 1,
                "batch_size": 8,
            },
            ppsci.loss.MSELoss("mean"),
            name="LAP",
        )
        solver = ppsci.solver.Solver(
            model,
            {"LAP": constraint},
            output_dir=str(tmp_path),
            optimizer=optimizer,
            epochs=1,
            iters_per_epoch=1,
            log_freq=1,
            device="cpu",
            equation={"laplace": equation},
            to_static=True,
        )

        solver.train()
        assert solver.global_step == 1
        assert core._is_all_prim_enabled()
    finally:
        core._set_prim_all_enabled(prev_prim)


if __name__ == "__main__":
    pytest.main()
