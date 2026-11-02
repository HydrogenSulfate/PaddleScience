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

import os

import numpy as np
import paddle
import pytest

import ppsci
from ppsci.utils import save_load


@pytest.fixture(autouse=True)
def _force_cpu_device():
    """Run every test on CPU and restore the previous device afterwards."""
    prev_device = paddle.device.get_device()
    paddle.device.set_device("cpu")
    paddle.seed(0)
    yield
    paddle.device.set_device(prev_device)


def _make_model():
    return ppsci.arch.MLP(("x",), ("u",), 2, 8)


def _make_optimizer(model):
    return ppsci.optimizer.Adam(1e-3)(model)


def _step_once(model, optimizer):
    """Run one tiny forward/backward/step so optimizer state is populated."""
    x = paddle.rand([4, 1])
    label = paddle.rand([4, 1])
    loss = paddle.nn.functional.mse_loss(model({"x": x})["u"], label)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()


def _state_to_numpy(model):
    return {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}


def _assert_state_allclose(model, expected):
    state = model.state_dict()
    assert set(state) == set(expected)
    for key, value in state.items():
        np.testing.assert_allclose(
            value.detach().numpy(), expected[key], rtol=1e-6, atol=1e-6
        )


def test_save_load_checkpoint_roundtrip(tmp_path):
    model = _make_model()
    optimizer = _make_optimizer(model)
    _step_once(model, optimizer)

    output_dir = str(tmp_path / "out")
    save_load.save_checkpoint(
        model,
        optimizer,
        {"metric": 0.25, "epoch": 3},
        output_dir=output_dir,
        prefix="model",
    )

    ckpt_dir = tmp_path / "out" / "checkpoints"
    for suffix in (".pdparams", ".pdopt", ".pdstates"):
        assert (ckpt_dir / f"model{suffix}").is_file()

    expected_state = _state_to_numpy(model)
    expected_optim_keys = set(optimizer.state_dict().keys())

    new_model = _make_model()
    new_optimizer = _make_optimizer(new_model)
    metric = save_load.load_checkpoint(
        str(ckpt_dir / "model"), new_model, new_optimizer
    )

    assert metric == {"metric": 0.25, "epoch": 3}
    _assert_state_allclose(new_model, expected_state)

    loaded_optim_state = new_optimizer.state_dict()
    assert len(loaded_optim_state) > 0
    assert set(loaded_optim_state.keys()) == expected_optim_keys


def test_save_checkpoint_model_only(tmp_path):
    model = _make_model()

    # metric=None -> only model params.
    none_dir = str(tmp_path / "none")
    save_load.save_checkpoint(
        model,
        None,
        None,
        output_dir=none_dir,
        prefix="model_none",
    )
    none_files = set(os.listdir(tmp_path / "none" / "checkpoints"))
    assert none_files == {"model_none.pdparams"}

    # metric={} -> still no .pdstates.
    empty_dir = str(tmp_path / "empty")
    save_load.save_checkpoint(
        model,
        None,
        {},
        output_dir=empty_dir,
        prefix="model_empty",
    )
    empty_files = set(os.listdir(tmp_path / "empty" / "checkpoints"))
    assert empty_files == {"model_empty.pdparams"}


def test_save_checkpoint_none_output_dir_is_noop(tmp_path):
    model = _make_model()
    save_load.save_checkpoint(model, None, None, output_dir=None)
    assert list(tmp_path.iterdir()) == []


def test_load_checkpoint_missing_pdparams_raises(tmp_path):
    model = _make_model()
    optimizer = _make_optimizer(model)
    with pytest.raises(FileNotFoundError):
        save_load.load_checkpoint(str(tmp_path / "does_not_exist"), model, optimizer)


def test_load_checkpoint_missing_pdopt_raises(tmp_path):
    model = _make_model()
    output_dir = str(tmp_path / "out")
    save_load.save_checkpoint(
        model, None, None, output_dir=output_dir, prefix="model_only"
    )

    ckpt_path = tmp_path / "out" / "checkpoints" / "model_only"
    assert ckpt_path.with_suffix(".pdparams").is_file()
    assert not ckpt_path.with_suffix(".pdopt").exists()

    new_model = _make_model()
    new_optimizer = _make_optimizer(new_model)
    with pytest.raises(FileNotFoundError):
        save_load.load_checkpoint(str(ckpt_path), new_model, new_optimizer)


def test_load_pretrain_roundtrip_and_pdparams_suffix(tmp_path):
    model = _make_model()
    output_dir = str(tmp_path / "out")
    save_load.save_checkpoint(
        model, None, None, output_dir=output_dir, prefix="pretrain"
    )
    expected_state = _state_to_numpy(model)
    ckpt_path = tmp_path / "out" / "checkpoints" / "pretrain"

    # Without the ".pdparams" suffix.
    model_without_suffix = _make_model()
    save_load.load_pretrain(model_without_suffix, str(ckpt_path))
    _assert_state_allclose(model_without_suffix, expected_state)

    # With the ".pdparams" suffix (exercises the suffix normalization).
    model_with_suffix = _make_model()
    save_load.load_pretrain(model_with_suffix, f"{ckpt_path}.pdparams")
    _assert_state_allclose(model_with_suffix, expected_state)


def test_load_pretrain_missing_path_raises(tmp_path):
    model = _make_model()
    with pytest.raises(FileNotFoundError):
        save_load.load_pretrain(model, str(tmp_path / "does_not_exist"))


def test_save_load_checkpoint_with_equation_learnable_params(tmp_path):
    pde = ppsci.equation.Vibration(1.0, 4.0, -1.0)
    assert len(pde.learnable_parameters) == 2

    model = _make_model()
    optimizer = _make_optimizer(model)
    output_dir = str(tmp_path / "out")
    save_load.save_checkpoint(
        model,
        optimizer,
        {"epoch": 1},
        output_dir=output_dir,
        prefix="vib",
        equation={"vib": pde},
    )

    ckpt_path = tmp_path / "out" / "checkpoints" / "vib"
    assert ckpt_path.with_suffix(".pdeqn").is_file()
    equation_dict = paddle.load(str(ckpt_path) + ".pdeqn")
    assert "vib" in equation_dict

    new_model = _make_model()
    new_optimizer = _make_optimizer(new_model)
    new_pde = ppsci.equation.Vibration(1.0, 0.0, 0.0)
    save_load.load_checkpoint(
        str(ckpt_path),
        new_model,
        new_optimizer,
        equation={"vib": new_pde},
    )

    saved_state = pde.state_dict()
    loaded_state = new_pde.state_dict()
    assert set(loaded_state) == set(saved_state)
    for key in saved_state:
        np.testing.assert_allclose(
            loaded_state[key].detach().numpy(),
            saved_state[key].detach().numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    # Saved values correspond to the constructor arguments k1=4.0, k2=-1.0.
    assert float(loaded_state["0"]) == pytest.approx(4.0, abs=1e-6)
    assert float(loaded_state["1"]) == pytest.approx(-1.0, abs=1e-6)


def test_load_pretrain_loads_pdeqn(tmp_path):
    pde = ppsci.equation.Vibration(1.0, 4.0, -1.0)
    model = _make_model()
    output_dir = str(tmp_path / "out")
    save_load.save_checkpoint(
        model,
        None,
        None,
        output_dir=output_dir,
        prefix="pre_vib",
        equation={"vib": pde},
    )

    ckpt_path = tmp_path / "out" / "checkpoints" / "pre_vib"
    expected_state = _state_to_numpy(model)

    new_model = _make_model()
    new_pde = ppsci.equation.Vibration(1.0, 0.0, 0.0)
    save_load.load_pretrain(new_model, str(ckpt_path), equation={"vib": new_pde})

    _assert_state_allclose(new_model, expected_state)
    loaded_state = new_pde.state_dict()
    assert float(loaded_state["0"]) == pytest.approx(4.0, abs=1e-6)
    assert float(loaded_state["1"]) == pytest.approx(-1.0, abs=1e-6)


def test_save_checkpoint_skips_pdeqn_without_learnable_params(tmp_path):
    # Laplace has no learnable parameters, so no .pdeqn sidecar is written.
    pde = ppsci.equation.Laplace(2)
    assert len(pde.learnable_parameters) == 0

    model = _make_model()
    output_dir = str(tmp_path / "out")
    save_load.save_checkpoint(
        model,
        None,
        None,
        output_dir=output_dir,
        prefix="no_vib",
        equation={"laplace": pde},
    )

    ckpt_dir = tmp_path / "out" / "checkpoints"
    assert (ckpt_dir / "no_vib.pdparams").is_file()
    assert not (ckpt_dir / "no_vib.pdeqn").exists()


if __name__ == "__main__":
    pytest.main()
