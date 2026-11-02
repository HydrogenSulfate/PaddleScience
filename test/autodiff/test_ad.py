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

from ppsci import autodiff


@pytest.fixture(autouse=True)
def reset_autodiff():
    """Keep the global jacobian/hessian caches isolated between tests."""
    autodiff.clear()
    yield
    autodiff.clear()


def test_jacobian_single_input_quadratic():
    x = paddle.randn([4, 1])
    x.stop_gradient = False
    y = x * x

    d = autodiff.jacobian(y, x)

    assert d.shape == [4, 1]
    assert bool(paddle.allclose(d, 2 * x))
    assert len(autodiff.jacobian.Js) == 1


def test_jacobian_list_of_inputs():
    x1 = paddle.randn([4, 1])
    x2 = paddle.randn([4, 1])
    x1.stop_gradient = False
    x2.stop_gradient = False
    y = x1 * x1 + 3 * x2

    js = autodiff.jacobian(y, [x1, x2])

    assert isinstance(js, list)
    assert len(js) == 2
    assert js[0].shape == [4, 1]
    assert js[1].shape == [4, 1]
    assert bool(paddle.allclose(js[0], 2 * x1))
    assert bool(paddle.allclose(js[1], 3 * paddle.ones([4, 1])))


def test_jacobian_component_selection():
    x = paddle.randn([4, 2])
    x.stop_gradient = False
    y = x * x

    # Cross components are zero for elementwise squares.
    assert bool(
        paddle.allclose(autodiff.jacobian(y, x, i=1, j=0), paddle.zeros([4, 1]))
    )
    assert bool(
        paddle.allclose(autodiff.jacobian(y, x, i=0, j=1), paddle.zeros([4, 1]))
    )

    j1 = autodiff.jacobian(y, x, i=1)
    assert j1.shape == [4, 2]
    assert bool(paddle.allclose(j1[:, 0:1], paddle.zeros([4, 1])))
    assert bool(paddle.allclose(j1[:, 1:2], 2 * x[:, 1:2]))


def test_hessian_quadratic():
    x = paddle.randn([4, 1])
    x.stop_gradient = False
    y = x * x

    h = autodiff.hessian(y, x)

    assert h.shape == [4, 1]
    assert bool(paddle.allclose(h, 2 * paddle.ones([4, 1])))
    # hessian internally uses jacobian, so only the Hessian cache count is
    # asserted here.
    assert len(autodiff.hessian.Hs) == 1


def test_hessian_component_validation():
    x = paddle.randn([4, 2])
    x.stop_gradient = False
    y = x * x

    with pytest.raises(ValueError, match="can not be None"):
        autodiff.hessian(y, x)
    with pytest.raises(ValueError, match="should be smaller than dim_y"):
        autodiff.hessian(y, x, component=2)

    x_single = paddle.randn([4, 1])
    x_single.stop_gradient = False
    with pytest.raises(ValueError, match="should be set to None"):
        autodiff.hessian(x_single * x_single, x_single, component=0)

    # `component` selects the output column; i/j select the input derivative
    # indices (both default to 0). y[:, c] is the sum of squares over inputs.
    s = (x * x).sum(axis=1, keepdim=True)
    y_sum = paddle.concat([s, s], axis=1)
    h = autodiff.hessian(y_sum, x, component=1)
    assert h.shape == [4, 1]
    assert bool(paddle.allclose(h, 2 * paddle.ones([4, 1])))


def test_clear_resets_caches():
    x = paddle.randn([4, 1])
    x.stop_gradient = False
    y = x * x

    autodiff.jacobian(y, x)
    autodiff.hessian(y, x)
    assert autodiff.jacobian.Js
    assert autodiff.hessian.Hs

    autodiff.clear()

    assert autodiff.jacobian.Js == {}
    assert autodiff.hessian.Hs == {}


def test_jacobian_index_out_of_range():
    x = paddle.randn([4, 2])
    x.stop_gradient = False
    y = x.sum(axis=1, keepdim=True)

    with pytest.raises(ValueError, match=r"i\(1\) should in range"):
        autodiff.jacobian(y, x, i=1)
    with pytest.raises(ValueError, match=r"j\(2\) should in range"):
        autodiff.jacobian(y, x, i=0, j=2)


if __name__ == "__main__":
    pytest.main()
