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

import subprocess
import sys


def _run(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_ppsci_is_lazy():
    _run(
        "import sys, ppsci; "
        "assert 'paddle' not in sys.modules, 'paddle should not be imported'; "
        "assert 'ppsci.arch' not in sys.modules, 'arch should not be imported'"
    )


def test_all_top_level_attrs_resolve():
    _run("import ppsci; [getattr(ppsci, name) for name in ppsci.__all__]")


def test_all_utils_attrs_resolve():
    _run(
        "import ppsci.utils; "
        "[getattr(ppsci.utils, name) for name in ppsci.utils.__all__]"
    )


def test_from_import_submodule_fallback():
    _run("from ppsci import externals, probability")


def test_omegaconf_resolvers_are_registered():
    _run(
        "from omegaconf import OmegaConf; "
        "import ppsci; "
        "assert OmegaConf.create({'x': '${sum:[1, 2, 3]}'}).x == 6; "
        "assert abs(OmegaConf.create({'x': '${numpy:pi}'}).x - 3.141592653589793) < 1e-12"
    )


def test_hydra_can_instantiate_configured_model():
    _run(
        "from hydra.utils import instantiate; "
        "from omegaconf import OmegaConf; "
        "import ppsci; "
        "cfg = OmegaConf.create({'_target_': 'ppsci.arch.MLP', "
        "'input_keys': ['x'], 'output_keys': ['u'], "
        "'num_layers': 2, 'hidden_size': 4}); "
        "model = instantiate(cfg); "
        "assert isinstance(model, ppsci.arch.MLP)"
    )


if __name__ == "__main__":
    import pytest

    pytest.main()
