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

import numpy as np
import paddle
import pytest

import ppsci
from ppsci.utils import expression


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


def _make_input(num_samples=4):
    return {
        "x": paddle.rand([num_samples, 1]),
        "y": paddle.rand([num_samples, 1]),
    }


def _make_expr_solver():
    return expression.ExpressionSolver()


def test_expression_solver_nvtx_flag_defaults_false():
    assert expression.ExpressionSolver().nvtx_flag is False


class _FakeConstraint:
    """Minimal stand-in exposing the attributes used by `train_forward`."""

    def __init__(self, losses):
        self.output_expr = {"u": lambda out: out["u"]}
        self._losses = losses

    def loss(self, output_dict, label_dict, weight_dict):
        # Keep stop_gradient=False so the autograd-retention assertion is meaningful.
        return {
            name: paddle.to_tensor(value, dtype="float32", stop_gradient=False)
            for name, value in self._losses.items()
        }


class _FakeValidator:
    def __init__(self, loss):
        self.loss = loss


def test_forward_raises_not_implemented():
    solver = expression.ExpressionSolver()
    with pytest.raises(NotImplementedError, match="train_forward"):
        solver.forward()


def test_train_forward_two_constraints_types_and_detached_grouping():
    model = _make_model()
    solver = _make_expr_solver()

    constraints = {
        "a": _FakeConstraint({"mse_u": 1.5, "mae_u": 0.5}),
        "b": _FakeConstraint({"mse_v": 2.0}),
    }
    input_dicts = (_make_input(), _make_input())

    losses_all, losses_constraint = solver.train_forward(
        tuple(cst.output_expr for cst in constraints.values()),
        input_dicts,
        model,
        constraints,
        ({}, {}),
        ({}, {}),
    )

    # Constraints (and their loss keys) are processed in dict insertion order.
    assert list(losses_all) == ["mse_u", "mae_u", "mse_v"]
    assert list(losses_constraint) == ["a", "b"]

    # Individual losses stay connected to the autograd graph for backward.
    for loss in losses_all.values():
        assert paddle.is_tensor(loss)
        assert loss.stop_gradient is False

    # Grouped losses are detached Python floats and equal the per-constraint sum.
    for value in losses_constraint.values():
        assert isinstance(value, float)
    assert losses_constraint["a"] == pytest.approx(1.5 + 0.5, abs=1e-5)
    assert losses_constraint["b"] == pytest.approx(2.0, abs=1e-5)

    # Sanity check that the flat losses keep their original values.
    assert float(losses_all["mse_u"]) == pytest.approx(1.5, abs=1e-6)
    assert float(losses_all["mse_v"]) == pytest.approx(2.0, abs=1e-6)


def test_train_forward_empty_losses_returns_zero_float():
    model = _make_model()
    solver = _make_expr_solver()

    constraints = {"empty": _FakeConstraint({})}
    input_dicts = (_make_input(),)

    losses_all, losses_constraint = solver.train_forward(
        tuple(cst.output_expr for cst in constraints.values()),
        input_dicts,
        model,
        constraints,
        ({},),
        ({},),
    )

    assert losses_all == {}
    assert losses_constraint == {"empty": 0.0}
    assert isinstance(losses_constraint["empty"], float)


def test_eval_forward_with_expr_and_callable_loss():
    model = _make_model()
    solver = _make_expr_solver()
    validator = _FakeValidator(
        loss=lambda output, label, weight: {"mse": paddle.to_tensor(0.5)}
    )
    input_dict = _make_input()

    output_dict, validator_losses = solver.eval_forward(
        {"u2": lambda out: out["u"] * 2},
        input_dict,
        model,
        validator,
        {},
        {},
    )

    assert set(output_dict) == {"u", "u2"}
    assert tuple(output_dict["u"].shape) == (4, 1)
    np.testing.assert_allclose(
        output_dict["u2"].numpy(), output_dict["u"].numpy() * 2, rtol=1e-6, atol=1e-6
    )

    assert set(validator_losses) == {"mse"}
    assert float(validator_losses["mse"]) == pytest.approx(0.5, abs=1e-6)


def test_eval_forward_non_callable_loss_returns_empty_dict():
    model = _make_model()
    solver = _make_expr_solver()
    validator = _FakeValidator(loss=None)
    input_dict = _make_input()

    output_dict, validator_losses = solver.eval_forward(
        {},
        input_dict,
        model,
        validator,
        {},
        {},
    )

    assert set(output_dict) == {"u"}
    assert validator_losses == {}


def test_visu_forward_none_expr_returns_model_outputs():
    model = _make_model()
    solver = _make_expr_solver()

    output_dict = solver.visu_forward(None, _make_input(), model)

    assert set(output_dict) == {"u"}
    assert tuple(output_dict["u"].shape) == (4, 1)


def test_visu_forward_with_expr_adds_keys():
    model = _make_model()
    solver = _make_expr_solver()

    output_dict = solver.visu_forward(
        {"u2": lambda out: out["u"] * 2},
        _make_input(),
        model,
    )

    assert set(output_dict) == {"u", "u2"}
    assert tuple(output_dict["u2"].shape) == (4, 1)
    np.testing.assert_allclose(
        output_dict["u2"].numpy(), output_dict["u"].numpy() * 2, rtol=1e-6, atol=1e-6
    )


if __name__ == "__main__":
    pytest.main()
