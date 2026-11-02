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

"""Tests for selected helpers in ppsci.utils.misc."""

import os
import random

import matplotlib

# Must be set before ppsci.utils.misc imports matplotlib.pyplot.
matplotlib.use("Agg")

import numpy as np
import paddle
import pytest

from ppsci.utils import misc


def test_average_meter_update_and_properties():
    meter = misc.AverageMeter("loss")
    meter.update(1, 2)
    meter.update(3, 2)

    assert meter.val == 3
    assert meter.sum == 8
    assert meter.count == 4
    assert meter.avg == 2.0
    assert meter.history == [1, 3]
    assert meter.avg_info == "loss: 2.00000"
    assert meter.total == "loss_sum: 8.000000"
    assert meter.mean == "loss: 2.000000"
    assert meter.value == "loss: 3.000000"


def test_average_meter_reset():
    meter = misc.AverageMeter("loss")
    meter.update(2)
    meter.reset()

    assert meter.val == 0
    assert meter.sum == 0
    assert meter.count == 0
    assert meter.avg == 0
    assert meter.history == []


def test_average_meter_tensor_input():
    meter = misc.AverageMeter("name")
    meter.update(paddle.to_tensor(2.0))

    assert isinstance(meter.avg, paddle.Tensor)
    assert meter.avg_info == "name: 2.00000"
    # avg_info coerces the stored tensor into a python float.
    assert isinstance(meter.avg, float)


def test_pretty_ordered_dict_str():
    data = misc.PrettyOrderedDict()
    data.update({"a": 1, "b": 2})

    assert str(data) == "('a', 1)('b', 2)"
    assert list(data.keys()) == ["a", "b"]


def test_pretty_defaultdict_str_and_default():
    data = misc.Prettydefaultdict(float)
    data.update({"a": 1, "b": 2})

    assert str(data) == "('a', 1)('b', 2)"
    assert data["missing"] == 0.0


def test_rank_zero_only_context():
    with misc.RankZeroOnly(0) as is_master:
        assert is_master is True
    with misc.RankZeroOnly(1) as is_master:
        assert is_master is False


def test_rank_zero_first_context():
    executed = []
    with misc.RankZeroFirst(0):
        executed.append(True)
    assert executed == [True]

    # In a non-distributed setting an explicit non-zero rank is ignored:
    # the constructor falls back to rank 0.
    with misc.RankZeroFirst(1):
        executed.append(True)
    assert executed == [True, True]
    assert misc.RankZeroFirst(1).is_master is True


def test_set_random_seed_reproducibility():
    misc.set_random_seed(123)
    with paddle.device_guard("cpu"):
        first = (
            np.random.rand(),
            random.random(),
            paddle.rand([3]).numpy(),
        )

    misc.set_random_seed(123)
    with paddle.device_guard("cpu"):
        second = (
            np.random.rand(),
            random.random(),
            paddle.rand([3]).numpy(),
        )

    assert first[0] == second[0]
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])


def test_run_at_rank0_executes_and_preserves_metadata():
    @misc.run_at_rank0
    def add_one(value):
        return value + 1

    assert add_one(4) == 5
    assert add_one.__name__ == "add_one"


def test_plot_curve_writes_file(tmp_path):
    misc.plot_curve(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]},
        xlabel="epoch",
        ylabel="loss",
        output_dir=str(tmp_path),
    )

    output_file = tmp_path / "epoch-loss_curve.jpg"
    assert output_file.exists()
    assert os.path.getsize(output_file) > 0


def test_plot_curve_smooth_step_and_nondivisible(tmp_path):
    misc.plot_curve(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0]},
        xlabel="epoch_smooth",
        ylabel="loss_smooth",
        output_dir=str(tmp_path),
        smooth_step=2,
    )

    output_file = tmp_path / "epoch_smooth-loss_smooth_curve.jpg"
    assert output_file.exists()
    assert os.path.getsize(output_file) > 0


def test_plot_curve_semilogy(tmp_path):
    misc.plot_curve(
        {"a": [1.0, 2.0, 3.0]},
        xlabel="epoch_log",
        ylabel="loss_log",
        output_dir=str(tmp_path),
        use_semilogy=True,
    )

    output_file = tmp_path / "epoch_log-loss_log_curve.jpg"
    assert output_file.exists()
    assert os.path.getsize(output_file) > 0


def test_convert_to_dict_and_array_roundtrip():
    array = np.arange(6, dtype=np.float32).reshape((2, 3))

    split = misc.convert_to_dict(array, ("x", "y", "z"))
    np.testing.assert_array_equal(split["x"].ravel(), [0.0, 3.0])
    np.testing.assert_array_equal(split["y"].ravel(), [1.0, 4.0])
    np.testing.assert_array_equal(split["z"].ravel(), [2.0, 5.0])

    merged = misc.convert_to_array(split, ("x", "y", "z"))
    np.testing.assert_array_equal(merged, array)

    # Wrong number of keys is only invalid for convert_to_dict.
    with pytest.raises(ValueError):
        misc.convert_to_dict(array, ("x", "y"))
    # Missing keys raise KeyError in convert_to_array.
    with pytest.raises(KeyError):
        misc.convert_to_array(split, ("x", "missing"))


def test_cartesian_product():
    # Inputs must be ndarrays: lists are treated as dtype specs internally.
    result = misc.cartesian_product(np.array([1, 2]), np.array([10, 20]))
    np.testing.assert_array_equal(result, [[1, 10], [1, 20], [2, 10], [2, 20]])


def test_concat_and_stack_dict_list():
    dic1 = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "y": np.array([[5.0, 6.0], [7.0, 8.0]]),
    }
    dic2 = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "y": np.array([[5.0, 6.0], [7.0, 8.0]]),
    }

    concatenated = misc.concat_dict_list((dic1, dic2))
    assert concatenated["x"].shape == (4, 2)
    assert concatenated["y"].shape == (4, 2)
    np.testing.assert_array_equal(concatenated["x"][2:], dic2["x"])

    stacked = misc.stack_dict_list((dic1, dic2))
    assert stacked["x"].shape == (2, 2, 2)
    assert stacked["y"].shape == (2, 2, 2)
    np.testing.assert_array_equal(stacked["x"][1], dic2["x"])


def test_combine_array_with_time():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = misc.combine_array_with_time(x, (1, 2))

    assert result.shape == (4, 3)
    np.testing.assert_array_equal(
        result,
        [
            [1.0, 1.0, 2.0],
            [1.0, 3.0, 4.0],
            [2.0, 1.0, 2.0],
            [2.0, 3.0, 4.0],
        ],
    )


def test_check_flag_enabled(monkeypatch):
    monkeypatch.delenv("PPS_TEST_FLAG", raising=False)
    assert misc.check_flag_enabled("PPS_TEST_FLAG") is False

    monkeypatch.setenv("PPS_TEST_FLAG", "1")
    assert misc.check_flag_enabled("PPS_TEST_FLAG") is True

    monkeypatch.setenv("PPS_TEST_FLAG", "true")
    assert misc.check_flag_enabled("PPS_TEST_FLAG") is True

    monkeypatch.setenv("PPS_TEST_FLAG", "0")
    assert misc.check_flag_enabled("PPS_TEST_FLAG") is False


if __name__ == "__main__":
    pytest.main()
