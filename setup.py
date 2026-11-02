"""Compatibility entry point for legacy ``python setup.py`` commands.

The canonical package metadata and build configuration live in ``pyproject.toml``.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
