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

from ppsci.arch.mlp import MLP
from ppsci.arch.mlp import ModifiedMLP
from ppsci.arch.mlp import RandomWeightFactorization


def test_mlp_forward_shapes_and_key_order():
    model = MLP(("x", "y"), ("u", "v"), 2, 8)
    out = model({"x": paddle.rand([4, 1]), "y": paddle.rand([4, 1])})

    assert tuple(out) == ("u", "v")
    for key in ("u", "v"):
        assert out[key].shape == [4, 1]
        assert out[key].dtype == paddle.float32
        assert bool(paddle.isfinite(out[key]).all())


def test_mlp_hidden_size_list():
    # hidden_size as a list requires num_layers=None.
    model = MLP(("x",), ("u",), None, [4, 8, 16])
    assert model({"x": paddle.rand([4, 1])})["u"].shape == [4, 1]

    # Tuple form is accepted as well.
    model = MLP(("x",), ("u",), None, (4, 8))
    assert model({"x": paddle.rand([4, 1])})["u"].shape == [4, 1]


def test_mlp_input_dim_output_dim():
    model = MLP(("x",), ("u", "v"), 2, 8, input_dim=4, output_dim=2)
    out = model({"x": paddle.rand([4, 4])})
    assert out["u"].shape == [4, 1]
    assert out["v"].shape == [4, 1]


def test_mlp_output_dim_split_across_output_keys():
    # output_dim is split evenly across output keys, so output_dim=6 with 3
    # output keys yields 2 columns per key.
    model = MLP(("x",), ("u", "v", "w"), 2, 8, output_dim=6)
    out = model({"x": paddle.rand([4, 1])})
    assert tuple(out) == ("u", "v", "w")
    for key in ("u", "v", "w"):
        assert out[key].shape == [4, 2]


def test_mlp_output_dim_not_divisible_raises():
    # paddle.split requires the output dimension to divide evenly by the number
    # of output keys; this documents the constraint enforced at forward time.
    model = MLP(("x",), ("u", "v", "w"), 2, 8, output_dim=7)
    with pytest.raises(ValueError, match="evenly divisible"):
        model({"x": paddle.rand([4, 1])})


def test_mlp_weight_norm():
    model = MLP(("x",), ("u",), 3, 8, weight_norm=True)
    out = model({"x": paddle.rand([4, 1])})

    assert out["u"].shape == [4, 1]
    assert bool(paddle.isfinite(out["u"]).all())
    assert model.num_params > 0
    # Hidden layers use WeightNormLinear, only last_fc is a plain nn.Linear.
    assert "num_fc = 1" in str(model)


def test_mlp_skip_connection():
    model = MLP(("x",), ("u",), 4, 8, skip_connection=True)
    assert model({"x": paddle.rand([4, 1])})["u"].shape == [4, 1]

    # Skip sources are hidden layers 0, 2, 4, ...; their widths must match.
    with pytest.raises(ValueError, match="skip_connection=True requires"):
        MLP(("x",), ("u",), None, [4, 8, 16], skip_connection=True)

    # Regression for the residual add itself. With identity linears/activations
    # and 6 hidden layers, the three skip points accumulate 1 -> 2 -> 4.
    model = MLP(("x",), ("u",), 6, 1, skip_connection=True)
    model.linears = paddle.nn.LayerList([paddle.nn.Identity() for _ in range(6)])
    model.acts = paddle.nn.LayerList([paddle.nn.Identity() for _ in range(6)])
    model.last_fc = paddle.nn.Identity()
    out = model({"x": paddle.ones([1, 1])})
    np.testing.assert_allclose(out["u"].numpy(), [[4.0]], rtol=1e-6)


def test_modified_mlp_skip_connection_residual():
    class _Ones(paddle.nn.Layer):
        def forward(self, x):
            return paddle.ones_like(x)

    class _Zeros(paddle.nn.Layer):
        def forward(self, x):
            return paddle.zeros_like(x)

    model = ModifiedMLP(("x",), ("u",), 6, 1, skip_connection=True)
    model.linears = paddle.nn.LayerList([paddle.nn.Identity() for _ in range(6)])
    model.acts = paddle.nn.LayerList([paddle.nn.Identity() for _ in range(6)])
    model.embed_u = _Ones()
    model.embed_v = _Zeros()
    model.last_fc = paddle.nn.Identity()

    out = model({"x": paddle.ones([1, 1])})
    np.testing.assert_allclose(out["u"].numpy(), [[4.0]], rtol=1e-6)


def test_mlp_fourier_embedding():
    # FourierEmbedding kernel has shape [in_features, dim // 2]; with two input
    # keys, in_features == 2.
    model = MLP(("x", "y"), ("u",), 2, 8, fourier={"dim": 8, "scale": 1.0})
    assert model.fourier_emb.kernel.shape == [2, 4]
    out = model({"x": paddle.rand([4, 1]), "y": paddle.rand([4, 1])})
    assert out["u"].shape == [4, 1]


def test_mlp_fourier_odd_dim_raises():
    with pytest.raises(ValueError, match="out_features must be even"):
        MLP(("x",), ("u",), 2, 8, fourier={"dim": 7, "scale": 1.0})


def test_mlp_random_weight():
    model = MLP(("x",), ("u",), 2, 8, random_weight={"mean": 0.5, "std": 0.1})
    out = model({"x": paddle.rand([4, 1])})

    assert out["u"].shape == [4, 1]
    assert isinstance(model.last_fc, RandomWeightFactorization)
    for param in model.parameters():
        assert bool(paddle.isfinite(param).all())


def test_mlp_input_output_transforms():
    model = MLP(("2x",), ("y",), 2, 8)

    def transform_in(in_):
        return {"2x": 2.0 * in_["x"]}

    def transform_out(in_, out):
        return {"y2": out["y"] + 1.0}

    model.register_input_transform(transform_in)
    model.register_output_transform(transform_out)

    out = model({"x": paddle.rand([4, 1])})
    assert tuple(out) == ("y2",)
    assert out["y2"].shape == [4, 1]


def test_mlp_periods_embedding():
    model = MLP(("x",), ("u",), 2, 8, periods={"x": (2.0, True)})
    input_dict = {"x": paddle.rand([4, 1])}
    out = model(input_dict)

    assert out["u"].shape == [4, 1]
    # frequency mu = 2*pi / period = pi for period=2.0
    assert np.isclose(float(model.period_emb.freqs_dict["x"]), np.pi)
    # PeriodEmbedding makes a shallow copy, so the caller's dict is unchanged.
    assert input_dict["x"].shape == [4, 1]

    # trainable=False freezes the frequency parameter.
    model_frozen = MLP(("x",), ("u",), 2, 8, periods={"x": (2.0, False)})
    assert bool(model_frozen.period_emb.freqs_dict["x"].stop_gradient)


def test_mlp_num_params_buffers_and_str():
    model = MLP(("x", "y"), ("u", "v"), 2, 8)

    # 2 hidden layers of 8 + output layer: (2*8+8) + (8*8+8) + (8*2+2) = 114.
    assert model.num_params == 114
    assert model.num_buffers == 0

    model_str = str(model)
    for fragment in (
        "MLP",
        "input_keys",
        "output_keys",
        "num_fc = 3",
        "num_conv = 0",
        "num_params = 114",
        "num_buffers = 0",
    ):
        assert fragment in model_str


def test_mlp_hidden_size_num_layers_mismatch():
    with pytest.raises(ValueError, match="num_layers should be None"):
        MLP(("x",), ("u",), 2, [8, 8])

    with pytest.raises(ValueError, match="num_layers should be an int"):
        MLP(("x",), ("u",), None, 8)

    with pytest.raises(ValueError, match="hidden_size should be list of int or int"):
        MLP(("x",), ("u",), 2, 8.0)


if __name__ == "__main__":
    pytest.main()
