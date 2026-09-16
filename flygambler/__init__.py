"""flygambler: a real fruit-fly connectome that learns to gamble via dopamine.

See the project README for the science and the honest caveats.
"""

# Brian2 still does a bare ``import distutils`` at import time. Python 3.12+
# removed stdlib distutils, and setuptools >= 74 no longer auto-activates its
# compatibility shim -- but importing setuptools first does activate it. Do
# that here so any ``from flygambler...`` import path fixes Brian2 up front.
import os as _os

_os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
try:  # pragma: no cover - environment shim
    import setuptools  # noqa: F401
except Exception:  # pragma: no cover
    pass

__all__ = ["connectome", "brain", "game", "simulate"]
