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
import pytest

from ppsci import constraint
from ppsci import geometry
from ppsci import loss

_BATCH_SIZE = 4
_ITERS = 2


def _dataloader_cfg(batch_size=_BATCH_SIZE, iters_per_epoch=_ITERS, **kwargs):
    return {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": iters_per_epoch,
        "batch_size": batch_size,
        # check_periodic_dataloader_cfg reads sampler["batch_size"], which is not
        # yet populated by build_dataloader at that point.
        "sampler": {"name": "BatchSampler", "batch_size": batch_size},
        **kwargs,
    }


def _build(geom=None, label_dict=None, **kwargs):
    if geom is None:
        geom = geometry.Rectangle((0, 0), (1, 1))
    if label_dict is None:
        label_dict = {"u": 0}
    return constraint.PeriodicConstraint(
        {"u": lambda out: out["u"]},
        label_dict,
        geom,
        "x",
        _dataloader_cfg(),
        loss.MSELoss("mean"),
        **kwargs,
    )


def test_periodic_input_length_and_keys():
    np.random.seed(0)
    con = _build()
    input_dict, _, _ = next(iter(con.data_iter))

    # batch_size * iters_per_epoch = 8 rows.
    for key in ("x", "y", "normal_x", "normal_y"):
        assert key in input_dict
        assert tuple(input_dict[key].shape) == (8, 1)


def test_periodic_output_keys_and_label_dict_attribute():
    # PeriodicConstraint derives output_keys from output_expr and stores
    # label_dict so the inherited __str__ works.
    np.random.seed(0)
    con = _build()
    assert con.output_keys == ("u",)
    assert tuple(con.output_expr.keys()) == ("u",)
    assert con.label_dict == {"u": 0}
    text = str(con)
    assert "PeriodicConstraint" in text
    assert "label_dict" in text


def test_periodic_blocks_contain_periodic_images():
    np.random.seed(0)
    geom = geometry.Rectangle((0, 0), (1, 1))
    con = _build(geom=geom)
    input_dict, _, _ = next(iter(con.data_iter))

    arr = {key: np.asarray(value) for key, value in input_dict.items()}
    component = geom.dim_keys.index("x")

    for block in range(_ITERS):
        start = block * _BATCH_SIZE
        orig = {key: arr[key][start : start + _BATCH_SIZE // 2] for key in arr}
        expected = geom.periodic_point(orig, component)
        for key in arr:
            np.testing.assert_allclose(
                arr[key][start + _BATCH_SIZE // 2 : start + _BATCH_SIZE],
                np.asarray(expected[key]),
            )


def test_periodic_labels_are_zeros():
    # build_periodic_label_dict ignores the supplied value and emits zeros.
    np.random.seed(0)
    con = _build(label_dict={"u": 123})
    _, label_dict, _ = next(iter(con.data_iter))
    assert tuple(label_dict["u"].shape) == (8, 1)
    np.testing.assert_allclose(np.asarray(label_dict["u"]), 0.0)


def test_periodic_odd_batch_size_raises():
    geom = geometry.Rectangle((0, 0), (1, 1))
    with pytest.raises(ValueError) as excinfo:
        constraint.PeriodicConstraint(
            {"u": lambda out: out["u"]},
            {"u": 0},
            geom,
            "x",
            _dataloader_cfg(batch_size=5),
            loss.MSELoss("mean"),
        )
    assert "batch_size(5)" in str(excinfo.value)
    assert "positive and even when using PeriodicConstraint" in str(excinfo.value)


def test_periodic_shuffle_true_raises():
    geom = geometry.Rectangle((0, 0), (1, 1))
    with pytest.raises(ValueError) as excinfo:
        constraint.PeriodicConstraint(
            {"u": lambda out: out["u"]},
            {"u": 0},
            geom,
            "x",
            _dataloader_cfg(shuffle=True),
            loss.MSELoss("mean"),
        )
    assert "shuffle(True)" in str(excinfo.value)
    assert "should be False when using PeriodicConstraint" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main()
