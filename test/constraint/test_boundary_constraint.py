# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import pytest
import sympy

from ppsci import constraint
from ppsci import geometry
from ppsci import loss


def _dataloader_cfg(batch_size=4, iters_per_epoch=1):
    return {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": iters_per_epoch,
        "batch_size": batch_size,
    }


def _build(output_expr=None, label_dict=None, **kwargs):
    rect = geometry.Rectangle((0, 0), (1, 1))
    if output_expr is None:
        output_expr = {"u": lambda out: out["u"]}
    if label_dict is None:
        label_dict = {"u": 0}
    return constraint.BoundaryConstraint(
        output_expr,
        label_dict,
        rect,
        _dataloader_cfg(batch_size=4, iters_per_epoch=1),
        loss.MSELoss("mean"),
        **kwargs,
    )


def test_boundary_input_and_output_keys():
    np.random.seed(0)
    bc = _build()
    assert bc.input_keys == ("x", "y")
    assert bc.output_keys == ("u",)

    input_dict, label_dict, _ = next(iter(bc.data_iter))
    for key in ("x", "y", "normal_x", "normal_y"):
        assert key in input_dict
        assert tuple(input_dict[key].shape) == (4, 1)
    assert tuple(label_dict["u"].shape) == (4, 1)
    np.testing.assert_allclose(np.asarray(label_dict["u"]), 0.0)


def test_boundary_output_expr_filters_unused_keys():
    np.random.seed(0)
    bc = _build(
        output_expr={"u": lambda out: out["u"], "unused": lambda out: out["u"] + 1},
        label_dict={"u": 0},
    )
    assert bc.output_keys == ("u",)
    assert set(bc.output_expr) == {"u"}


@pytest.mark.parametrize("value", [0, 2.5])
def test_boundary_numeric_labels(value):
    np.random.seed(0)
    bc = _build(label_dict={"u": value})
    input_dict, label_dict, _ = next(iter(bc.data_iter))
    assert tuple(label_dict["u"].shape) == (4, 1)
    assert np.asarray(label_dict["u"]).dtype == np.asarray(input_dict["x"]).dtype
    np.testing.assert_allclose(np.asarray(label_dict["u"]), value)


def test_boundary_sympy_label():
    np.random.seed(0)
    expr = sympy.Symbol("x") + 2 * sympy.Symbol("y")
    bc = _build(label_dict={"u": expr})
    input_dict, label_dict, _ = next(iter(bc.data_iter))
    np.testing.assert_allclose(
        np.asarray(label_dict["u"]),
        np.asarray(input_dict["x"]) + 2 * np.asarray(input_dict["y"]),
        rtol=1e-5,
    )


def test_boundary_callable_label():
    np.random.seed(0)
    bc = _build(label_dict={"u": lambda d: d["x"] * 3})
    input_dict, label_dict, _ = next(iter(bc.data_iter))
    np.testing.assert_allclose(
        np.asarray(label_dict["u"]), np.asarray(input_dict["x"]) * 3
    )

    bc_scalar = _build(label_dict={"u": lambda d: 7})
    _, label_dict_scalar, _ = next(iter(bc_scalar.data_iter))
    assert tuple(label_dict_scalar["u"].shape) == (4, 1)
    np.testing.assert_allclose(np.asarray(label_dict_scalar["u"]), 7.0)


@pytest.mark.parametrize("value", [2, 0.5])
def test_boundary_numeric_weights(value):
    np.random.seed(0)
    bc = _build(weight_dict={"u": value})
    _, _, weight_dict = next(iter(bc.data_iter))
    assert tuple(weight_dict["u"].shape) == (4, 1)
    np.testing.assert_allclose(np.asarray(weight_dict["u"]), value)


def test_boundary_sympy_weight():
    np.random.seed(0)
    expr = sympy.Symbol("x") + sympy.Symbol("y")
    bc = _build(weight_dict={"u": expr})
    input_dict, _, weight_dict = next(iter(bc.data_iter))
    np.testing.assert_allclose(
        np.asarray(weight_dict["u"]),
        np.asarray(input_dict["x"]) + np.asarray(input_dict["y"]),
        rtol=1e-5,
    )


def test_boundary_callable_weight():
    np.random.seed(0)
    bc = _build(weight_dict={"u": lambda d: d["y"] * 2})
    input_dict, _, weight_dict = next(iter(bc.data_iter))
    np.testing.assert_allclose(
        np.asarray(weight_dict["u"]), np.asarray(input_dict["y"]) * 2
    )

    bc_scalar = _build(weight_dict={"u": lambda d: 3})
    _, _, weight_scalar = next(iter(bc_scalar.data_iter))
    assert tuple(weight_scalar["u"].shape) == (4, 1)
    np.testing.assert_allclose(np.asarray(weight_scalar["u"]), 3.0)


def test_boundary_string_criteria_resolved():
    np.random.seed(0)
    bc = _build(criteria="lambda x, y: x >= 1.0")
    input_dict, _, _ = next(iter(bc.data_iter))
    # Only points on the right edge (x == 1.0) survive the filter.
    np.testing.assert_allclose(np.asarray(input_dict["x"]), 1.0)
    assert tuple(input_dict["x"].shape) == (4, 1)


def test_boundary_area_scaled_by_iters_per_epoch():
    np.random.seed(0)

    class _AreaGeom:
        dim_keys = ("x", "y")

        def sample_boundary(self, n, random, criteria, evenly):
            return {
                "x": np.zeros((n, 1), dtype=paddle.get_default_dtype()),
                "y": np.zeros((n, 1), dtype=paddle.get_default_dtype()),
                "area": np.full((n, 1), 3.0, dtype=paddle.get_default_dtype()),
            }

    bc = constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        _AreaGeom(),
        _dataloader_cfg(batch_size=4, iters_per_epoch=2),
        loss.MSELoss("mean"),
    )
    input_dict, _, _ = next(iter(bc.data_iter))
    # sample_boundary returns n=8 rows with area 3.0, then _scale_area
    # multiplies it by iters_per_epoch (2).
    assert tuple(input_dict["area"].shape) == (8, 1)
    np.testing.assert_allclose(np.asarray(input_dict["area"]), 6.0)


def test_boundary_invalid_string_weight_raises():
    # BoundaryConstraint calls build_weight_dict with allow_sdf=False, so a
    # string weight falls through to the generic invalid-type branch.
    with pytest.raises(NotImplementedError) as excinfo:
        _build(weight_dict={"u": "sdf"})
    assert str(excinfo.value) == "type of <class 'str'> is invalid yet."


if __name__ == "__main__":
    pytest.main()
