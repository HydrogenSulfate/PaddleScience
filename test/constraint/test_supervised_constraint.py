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
from ppsci import loss


def _data():
    x = np.arange(8, dtype="float32").reshape(8, 1)
    u = (x * 2).astype("float32")
    return {"x": x}, {"u": u}


def test_supervised_named_array_output_expr_none():
    input_dict, label_dict = _data()
    sc = constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict,
                "label": label_dict,
            },
            "iters_per_epoch": 1,
            "batch_size": 4,
        },
        loss.MSELoss("mean"),
    )

    assert sc.output_keys == tuple(label_dict.keys()) == ("u",)
    assert set(sc.output_expr) == {"u"}
    assert callable(sc.output_expr["u"])

    batch_input, batch_label, batch_weight = next(iter(sc.data_iter))
    np.testing.assert_allclose(np.asarray(batch_input["x"]), input_dict["x"][:4])
    np.testing.assert_allclose(np.asarray(batch_label["u"]), label_dict["u"][:4])
    assert batch_weight == {}
    # The auto-generated expression simply selects the label key.
    np.testing.assert_allclose(
        np.asarray(sc.output_expr["u"](batch_label)), np.asarray(batch_label["u"])
    )


def test_supervised_iterable_output_expr_none():
    input_dict, label_dict = _data()
    sc = constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": input_dict,
                "label": label_dict,
            },
            "iters_per_epoch": 1,
            "batch_size": 4,
        },
        loss.MSELoss("mean"),
    )

    assert sc.output_keys == ("u",)
    assert set(sc.output_expr) == {"u"}

    batch_input, batch_label, batch_weight = next(iter(sc.data_iter))
    # Iterable datasets yield the full data in a single batch.
    assert tuple(np.asarray(batch_input["x"]).shape) == (8, 1)
    assert tuple(np.asarray(batch_label["u"]).shape) == (8, 1)
    assert batch_weight is None


def test_supervised_explicit_output_expr():
    input_dict, label_dict = _data()
    sc = constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict,
                "label": label_dict,
            },
            "iters_per_epoch": 1,
            "batch_size": 4,
        },
        loss.MSELoss("mean"),
        output_expr={"v": lambda out: out["u"] * 2},
    )

    # Explicit keys drive output_keys instead of dataset.label_keys.
    assert sc.output_keys == ("v",)
    assert callable(sc.output_expr["v"])

    _, batch_label, _ = next(iter(sc.data_iter))
    np.testing.assert_allclose(
        np.asarray(sc.output_expr["v"](batch_label)),
        np.asarray(batch_label["u"]) * 2,
    )


def test_supervised_str_contains_fields():
    input_dict, label_dict = _data()
    sc = constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict,
                "label": label_dict,
            },
            "iters_per_epoch": 1,
            "batch_size": 4,
        },
        loss.MSELoss("mean"),
        name="sup_test",
    )
    text = str(sc)
    assert text.startswith("SupervisedConstraint")
    for field in ("name", "input_keys", "output_keys", "output_expr", "loss"):
        assert field in text
    assert "sup_test" in text
    assert "label_dict" not in text


if __name__ == "__main__":
    pytest.main()
