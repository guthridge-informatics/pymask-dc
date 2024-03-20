"""
.. module:: pymask_dc
   :platform: Unix, Windows
   :synopsis: Commandline interface for generating masks using Deepcell Mesmer

.. moduleauthor:: Miles Smith <miles-smith@omrf.org>
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

