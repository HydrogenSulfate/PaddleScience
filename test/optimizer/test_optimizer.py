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

import paddle
import pytest

from ppsci.optimizer import LBFGS
from ppsci.optimizer import SGD
from ppsci.optimizer import Adam
from ppsci.optimizer import AdamW
from ppsci.optimizer import Momentum
from ppsci.optimizer import OptimizerList
from ppsci.optimizer import build_optimizer


@pytest.fixture(autouse=True)
def _force_cpu_device():
    prev_device = paddle.device.get_device()
    paddle.device.set_device("cpu")
    paddle.seed(0)
    yield
    paddle.device.set_device(prev_device)


def _make_model():
    # Linear(2, 4) + Linear(4, 1) -> 4 parameters, 17 elements in total.
    return paddle.nn.Sequential(
        paddle.nn.Linear(2, 4),
        paddle.nn.Linear(4, 1),
    )


def _assert_param_shape(optimizer):
    assert len(optimizer._parameter_list) == 4
    assert int(sum(p.numel() for p in optimizer._parameter_list)) == 17


def test_sgd_optimizer():
    optimizer = SGD(0.01)(_make_model())
    assert isinstance(optimizer, paddle.optimizer.SGD)
    assert optimizer.get_lr() == pytest.approx(0.01)
    _assert_param_shape(optimizer)


def test_momentum_optimizer():
    optimizer = Momentum(0.02, 0.9)(_make_model())
    assert isinstance(optimizer, paddle.optimizer.Momentum)
    assert optimizer.get_lr() == pytest.approx(0.02)
    _assert_param_shape(optimizer)


def test_adam_optimizer():
    optimizer = Adam(0.03)(_make_model())
    assert isinstance(optimizer, paddle.optimizer.Adam)
    assert optimizer.get_lr() == pytest.approx(0.03)
    _assert_param_shape(optimizer)


def test_adamw_optimizer():
    model = _make_model()
    wrapper = AdamW(0.04, no_weight_decay_name="bias")
    optimizer = wrapper(model)

    assert isinstance(optimizer, paddle.optimizer.AdamW)
    assert optimizer.get_lr() == pytest.approx(0.04)
    # NOTE: use canonical parameter names (p.name), not named_parameters keys.
    for param in model.parameters():
        is_weight = param.name.endswith("w_0")
        assert wrapper._apply_decay_param_fun(param.name) is is_weight


def test_optimizer_weight_decay_and_grad_clip_step():
    model = _make_model()
    optimizer = SGD(
        1e-2,
        weight_decay=1e-4,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(0.5),
    )(model)

    before = [param.clone() for param in model.parameters()]
    output = model(paddle.ones((3, 2), dtype="float32"))
    loss = (output**2).mean()
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    assert any(
        not bool(paddle.equal_all(old, new).item())
        for old, new in zip(before, model.parameters())
    )
    assert optimizer.get_lr() == pytest.approx(1e-2)
    # clear_grad zeroes gradients (Paddle keeps zero tensors rather than None).
    for param in model.parameters():
        assert param.grad is not None
        assert float(param.grad.abs().max()) == pytest.approx(0.0)


def test_optimizer_list():
    model = _make_model()
    adam = Adam(1e-2)(model)
    sgd = SGD(1e-3)(model)
    optimizer_list = OptimizerList((adam, sgd))

    assert len(optimizer_list) == 2
    assert optimizer_list.get_lr() == pytest.approx(1e-2)
    assert optimizer_list[0] is adam
    assert list(optimizer_list) == [adam, sgd]

    state_dicts = optimizer_list.state_dict()
    assert len(state_dicts) == 2
    optimizer_list.set_state_dict(state_dicts)

    output = model(paddle.ones((2, 2), dtype="float32"))
    (output**2).mean().backward()
    optimizer_list.step()
    optimizer_list.clear_grad()


def test_optimizer_list_rejects_lbfgs():
    model = _make_model()
    with pytest.raises(ValueError, match="LBFGS is not supported"):
        OptimizerList((LBFGS()(model),))


def test_build_optimizer_with_float_lr():
    model = _make_model()
    optimizer, scheduler = build_optimizer(
        {"name": "Adam", "lr": 1e-2}, (model,), 10, 5
    )

    assert isinstance(optimizer, paddle.optimizer.Adam)
    assert optimizer.get_lr() == pytest.approx(1e-2)
    assert scheduler is None


def test_build_optimizer_with_lr_scheduler():
    model = _make_model()
    optimizer, scheduler = build_optimizer(
        {
            "name": "SGD",
            # NOTE: build_lr_scheduler uses eval on the dotted name; a bare
            # "Linear" raises NameError.
            "lr": {"name": "lr_scheduler.Linear", "learning_rate": 1e-2},
        },
        (model,),
        10,
        5,
    )

    assert isinstance(optimizer, paddle.optimizer.SGD)
    assert isinstance(scheduler, paddle.optimizer.lr.PolynomialDecay)
    assert scheduler.get_lr() == pytest.approx(1e-2)
    assert scheduler.by_epoch is False


if __name__ == "__main__":
    pytest.main()
