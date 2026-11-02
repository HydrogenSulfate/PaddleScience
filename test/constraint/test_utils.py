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

from ppsci.constraint import utils
from ppsci.data.dataset import NamedArrayDataset


class _FakeGeom:
    """Minimal geometry exposing only what the utils helpers consume."""

    dim_keys = ("x", "y")


def _input_dict(seed=0):
    rng = np.random.RandomState(seed)
    return {
        "x": rng.rand(4, 1).astype(paddle.get_default_dtype()),
        "y": rng.rand(4, 1).astype(paddle.get_default_dtype()),
    }


# --------------------------------------------------------------------------- #
# resolve_criteria
# --------------------------------------------------------------------------- #
def test_resolve_criteria_callable_passthrough():
    func = lambda x: x > 0.5  # noqa: E731
    assert utils.resolve_criteria(func) is func
    assert utils.resolve_criteria(None) is None


def test_resolve_criteria_string_uses_caller_namespace():
    # Self-contained expression.
    f = utils.resolve_criteria("lambda x: x > 0.5", None, None)
    assert f(np.array([0.1, 0.9])).tolist() == [False, True]

    # Free variables provided through eval_globals work because the function
    # closure keeps a reference to the globals mapping.
    g = utils.resolve_criteria("lambda x: x > threshold", {"threshold": 0.5}, None)
    assert g(np.array([0.1, 0.9])).tolist() == [False, True]

    # A non-function expression can use eval_locals directly.
    assert utils.resolve_criteria("threshold * 2", None, {"threshold": 0.5}) == 1.0

    # Modules can be injected through eval_globals.
    arr = utils.resolve_criteria("np.array([1, 2])", {"np": np}, None)
    np.testing.assert_array_equal(arr, np.array([1, 2]))


# --------------------------------------------------------------------------- #
# _scale_area
# --------------------------------------------------------------------------- #
def test_scale_area_multiplies_in_place():
    area = np.ones((4, 1), dtype=paddle.get_default_dtype())
    data = {"area": area}
    utils._scale_area(data, 2)
    assert data["area"] is area
    np.testing.assert_allclose(area, 2.0)


def test_scale_area_without_area_is_noop():
    x = np.ones((4, 1), dtype=paddle.get_default_dtype())
    data = {"x": x}
    utils._scale_area(data, 3)
    assert set(data) == {"x"}
    np.testing.assert_allclose(data["x"], 1.0)


# --------------------------------------------------------------------------- #
# build_label_dict
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [0, 2.5])
def test_build_label_dict_numeric(value):
    input_dict = _input_dict()
    label = utils.build_label_dict({"u": value}, input_dict, _FakeGeom())
    np.testing.assert_allclose(label["u"], value)
    assert label["u"].shape == input_dict["x"].shape
    assert label["u"].dtype == input_dict["x"].dtype


def test_build_label_dict_sympy():
    input_dict = _input_dict()
    expr = sympy.Symbol("x") + 2 * sympy.Symbol("y")
    label = utils.build_label_dict({"u": expr}, input_dict, _FakeGeom())
    np.testing.assert_allclose(
        label["u"], input_dict["x"] + 2 * input_dict["y"], rtol=1e-5
    )


def test_build_label_dict_callable():
    input_dict = _input_dict()

    array_label = utils.build_label_dict(
        {"u": lambda d: d["x"] * 3}, input_dict, _FakeGeom()
    )
    np.testing.assert_allclose(array_label["u"], input_dict["x"] * 3)

    scalar_label = utils.build_label_dict({"u": lambda d: 7}, input_dict, _FakeGeom())
    assert scalar_label["u"].shape == input_dict["x"].shape
    np.testing.assert_allclose(scalar_label["u"], 7.0)


def test_build_label_dict_invalid_type_raises():
    with pytest.raises(
        NotImplementedError, match=r"type of <class 'str'> is invalid yet\."
    ):
        utils.build_label_dict({"u": "not_a_number"}, _input_dict(), _FakeGeom())


# --------------------------------------------------------------------------- #
# build_weight_dict
# --------------------------------------------------------------------------- #
def test_build_weight_dict_none():
    input_dict = _input_dict()
    label = utils.build_label_dict({"u": 0}, input_dict, _FakeGeom())
    assert utils.build_weight_dict(None, label, input_dict, _FakeGeom()) is None


def test_build_weight_dict_default_ones():
    input_dict = _input_dict()
    label = {"u": np.full_like(input_dict["x"], 0.0)}
    weight = utils.build_weight_dict({}, label, input_dict, _FakeGeom())
    np.testing.assert_allclose(weight["u"], np.ones_like(label["u"]))
    assert weight["u"].shape == label["u"].shape


@pytest.mark.parametrize("value", [2, 0.5])
def test_build_weight_dict_numeric(value):
    input_dict = _input_dict()
    label = {"u": np.full_like(input_dict["x"], 0.0)}
    weight = utils.build_weight_dict({"u": value}, label, input_dict, _FakeGeom())
    np.testing.assert_allclose(weight["u"], value)
    assert weight["u"].shape == label["u"].shape


def test_build_weight_dict_sympy():
    input_dict = _input_dict()
    label = {"u": np.full_like(input_dict["x"], 0.0)}
    expr = sympy.Symbol("x") + sympy.Symbol("y")
    weight = utils.build_weight_dict({"u": expr}, label, input_dict, _FakeGeom())
    np.testing.assert_allclose(
        weight["u"], input_dict["x"] + input_dict["y"], rtol=1e-5
    )


def test_build_weight_dict_callable():
    input_dict = _input_dict()
    label = {"u": np.full_like(input_dict["x"], 0.0)}

    array_weight = utils.build_weight_dict(
        {"u": lambda d: d["y"] * 2}, label, input_dict, _FakeGeom()
    )
    np.testing.assert_allclose(array_weight["u"], input_dict["y"] * 2)

    scalar_weight = utils.build_weight_dict(
        {"u": lambda d: 3}, label, input_dict, _FakeGeom()
    )
    assert scalar_weight["u"].shape == input_dict["x"].shape
    np.testing.assert_allclose(scalar_weight["u"], 3.0)


def test_build_weight_dict_allow_sdf():
    input_dict = _input_dict()
    input_dict["sdf"] = np.ones((4, 1), dtype=paddle.get_default_dtype())
    label = {"u": np.full_like(input_dict["x"], 0.0)}
    weight = utils.build_weight_dict(
        {"u": "sdf"}, label, input_dict, _FakeGeom(), allow_sdf=True
    )
    assert weight["u"] is input_dict["sdf"]


def test_build_weight_dict_invalid_type_raises():
    input_dict = _input_dict()
    label = {"u": np.full_like(input_dict["x"], 0.0)}
    with pytest.raises(
        NotImplementedError, match=r"type of <class 'str'> is invalid yet\."
    ):
        utils.build_weight_dict({"u": "sdf"}, label, input_dict, _FakeGeom())


# --------------------------------------------------------------------------- #
# build_constraint_dataset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dataset", ["NamedArrayDataset", {"name": "NamedArrayDataset"}]
)
def test_build_constraint_dataset_does_not_mutate_cfg(dataset):
    input_dict = _input_dict()
    label = {"u": np.full_like(input_dict["x"], 0.0)}
    dataloader_cfg = {"dataset": dataset, "batch_size": 4}
    original_dataset_cfg = dataloader_cfg["dataset"]

    ds = utils.build_constraint_dataset(input_dict, label, None, dataloader_cfg)

    assert isinstance(ds, NamedArrayDataset)
    assert dataloader_cfg["dataset"] is original_dataset_cfg
    assert dataloader_cfg["dataset"] == original_dataset_cfg


def test_build_constraint_dataset_returns_named_array_dataset():
    input_dict = _input_dict()
    label = {"u": np.full_like(input_dict["x"], 0.0)}
    weight = {"u": np.ones_like(label["u"])}
    dataloader_cfg = {"dataset": "NamedArrayDataset", "batch_size": 4}

    ds = utils.build_constraint_dataset(input_dict, label, weight, dataloader_cfg)

    assert isinstance(ds, NamedArrayDataset)
    assert ds.input_keys == tuple(input_dict.keys())
    assert ds.label_keys == tuple(label.keys())
    assert set(ds.weight) == set(weight)
    for key in input_dict:
        np.testing.assert_array_equal(ds.input[key], input_dict[key])
    for key in label:
        np.testing.assert_array_equal(ds.label[key], label[key])
    for key in weight:
        np.testing.assert_array_equal(ds.weight[key], weight[key])
    # build_dataset deep-copies the config, so the dataset does not alias the
    # caller's arrays.
    assert ds.input is not input_dict


if __name__ == "__main__":
    pytest.main()
