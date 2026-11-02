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

from ppsci.arch import activation as act_mod


@pytest.mark.parametrize(
    "act_name",
    [
        "relu",
        "tanh",
        "sigmoid",
        "elu",
        "selu",
        "gelu",
        "leaky_relu",
        "silu",
        "sin",
        "cos",
        "swish",
        "identity",
        "siren",
    ],
)
def test_get_activation_known_names(act_name):
    # "stan" is deliberately excluded: get_activation("stan") returns the class
    # rather than an instance, so it is covered separately.
    act_func = act_mod.get_activation(act_name)
    x = paddle.rand([2, 4])
    y = act_func(x)

    assert callable(act_func)
    assert isinstance(y, paddle.Tensor)
    assert y.shape == [2, 4]
    assert y.dtype == paddle.float32


def test_get_activation_case_insensitive():
    x = paddle.to_tensor([[0.1, -0.2, 0.3, 0.4]])

    tanh = act_mod.get_activation("tanh")
    tanh_lower = act_mod.get_activation("Tanh")
    tanh_upper = act_mod.get_activation("TANH")
    assert bool(paddle.allclose(tanh(x), tanh_lower(x)))
    assert bool(paddle.allclose(tanh(x), tanh_upper(x)))

    # Other case variants return callable instances too.
    assert callable(act_mod.get_activation("ReLU"))


def test_get_activation_stan_returns_class():
    # Documented special case: "stan" returns the class, not an instance.
    stan_cls = act_mod.get_activation("stan")
    assert stan_cls is act_mod.Stan
    assert isinstance(stan_cls, type)

    x = paddle.rand([2, 4])
    y = stan_cls(4)(x)
    assert y.shape == [2, 4]
    assert y.dtype == paddle.float32


def test_get_activation_unknown_raises():
    with pytest.raises(ValueError, match="not found in act_func_dict"):
        act_mod.get_activation("not_a_real_act")


def test_activation_identity_and_relu_values():
    x = paddle.to_tensor([[-1.0, 0.0, 2.0, 3.0]])

    identity = act_mod.get_activation("identity")
    relu = act_mod.get_activation("relu")

    assert bool(paddle.allclose(identity(x), x))
    assert bool(paddle.allclose(relu(x), paddle.clip(x, min=0)))


if __name__ == "__main__":
    pytest.main()
