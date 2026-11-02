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

from ppsci.data.process.transform import Scale
from ppsci.data.process.transform import Translate


def test_scale_does_not_mutate_input():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    input_dict = {"x": x, "y": y}
    label_dict = {"u": np.array([1.0])}
    weight_dict = {"u": np.array([2.0])}

    scaled, out_label, out_weight = Scale({"x": 2.0, "missing": 3.0})(
        input_dict, label_dict, weight_dict
    )

    np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(y, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(scaled["x"], np.array([2.0, 4.0, 6.0]))
    np.testing.assert_array_equal(scaled["y"], y)
    assert out_label is label_dict
    assert out_weight is weight_dict


def test_translate_does_not_mutate_input():
    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    translated, _, _ = Translate({"x": 10.0, "missing": 1.0})(input_dict, {}, {})

    np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(translated["x"], np.array([11.0, 12.0, 13.0]))


def test_scale_and_translate_scalar_inputs():
    input_dict = {"x": 10, "y": 20}

    scaled, _, _ = Scale({"x": 1.5})(input_dict, {}, {})
    assert scaled == {"x": 15.0, "y": 20}
    assert input_dict == {"x": 10, "y": 20}

    translated, _, _ = Translate({"x": -1.0})(input_dict, {}, {})
    assert translated == {"x": 9.0, "y": 20}
    assert input_dict == {"x": 10, "y": 20}


if __name__ == "__main__":
    pytest.main()
