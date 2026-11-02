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

import paddle
import pytest

import ppsci


def test_model_list_output_keys_order():
    model1 = ppsci.arch.MLP(("x", "y"), ("u", "v", "w"), 2, 8)
    model2 = ppsci.arch.MLP(("x", "y"), ("p", "q", "r", "s", "t"), 2, 8)
    model = ppsci.arch.ModelList((model1, model2))

    # Keys follow the model order, and each model's own output order.
    assert model.input_keys == ("x", "y")
    assert model.output_keys == ("u", "v", "w", "p", "q", "r", "s", "t")

    output_dict = model({"x": paddle.rand([4, 1]), "y": paddle.rand([4, 1])})
    assert tuple(output_dict) == ("u", "v", "w", "p", "q", "r", "s", "t")


def test_model_list_input_keys_order_and_dedup():
    model1 = ppsci.arch.MLP(("y", "x"), ("u",), 2, 8)
    model2 = ppsci.arch.MLP(("x", "z"), ("v",), 2, 8)
    model = ppsci.arch.ModelList((model1, model2))

    # First-seen order is kept and duplicate "x" is removed.
    assert model.input_keys == ("y", "x", "z")


def test_model_list_duplicate_output_keys_raises():
    model1 = ppsci.arch.MLP(("x",), ("u",), 2, 8)
    model2 = ppsci.arch.MLP(("x",), ("u",), 2, 8)

    with pytest.raises(ValueError, match="got duplicate keys"):
        ppsci.arch.ModelList((model1, model2))


if __name__ == "__main__":
    pytest.main()
