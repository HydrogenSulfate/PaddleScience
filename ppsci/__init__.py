# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List

from ppsci.utils.config import apply_config as _apply_config

try:
    # import auto-generated version information from '._version' file, using
    # setuptools_scm via 'pip install'. Details of versioning rule can be referd to:
    # https://peps.python.org/pep-0440/#public-version-identifiers
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"

if TYPE_CHECKING:
    from ppsci import arch
    from ppsci import autodiff
    from ppsci import constraint
    from ppsci import data
    from ppsci import equation
    from ppsci import experimental
    from ppsci import geometry
    from ppsci import loss
    from ppsci import metric
    from ppsci import optimizer
    from ppsci import solver
    from ppsci import utils
    from ppsci import validate
    from ppsci import visualize
    from ppsci.utils import misc
    from ppsci.utils.checker import run_check
    from ppsci.utils.checker import run_check_mesh
    from ppsci.utils.symbolic import lambdify

__all__ = [
    "arch",
    "autodiff",
    "constraint",
    "data",
    "equation",
    "geometry",
    "loss",
    "metric",
    "optimizer",
    "utils",
    "visualize",
    "validate",
    "solver",
    "experimental",
    "run_check",
    "run_check_mesh",
    "lambdify",
    "misc",
]

# name -> submodule to import on first attribute access
_SUBMODULES: Dict[str, str] = {
    "arch": "ppsci.arch",
    "autodiff": "ppsci.autodiff",
    "constraint": "ppsci.constraint",
    "data": "ppsci.data",
    "equation": "ppsci.equation",
    "experimental": "ppsci.experimental",
    "geometry": "ppsci.geometry",
    "loss": "ppsci.loss",
    "metric": "ppsci.metric",
    "optimizer": "ppsci.optimizer",
    "solver": "ppsci.solver",
    "utils": "ppsci.utils",
    "validate": "ppsci.validate",
    "visualize": "ppsci.visualize",
}

# name -> (module to import, attribute to fetch; None means the module itself)
_ATTRS: Dict[str, tuple] = {
    "lambdify": ("ppsci.utils.symbolic", "lambdify"),
    "misc": ("ppsci.utils.misc", None),
    "run_check": ("ppsci.utils.checker", "run_check"),
    "run_check_mesh": ("ppsci.utils.checker", "run_check_mesh"),
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        value = importlib.import_module(_SUBMODULES[name])
    elif name in _ATTRS:
        module_name, attr_name = _ATTRS[name]
        module = importlib.import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


# Initialize shared configuration once at the package entry point, before Hydra
# composes example configs. Importing config.py itself does not register them.
_apply_config()
