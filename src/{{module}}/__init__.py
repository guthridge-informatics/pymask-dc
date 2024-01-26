"""
.. module:: {{module}}
   :platform: Unix, Windows
   :synopsis: {{description}}

.. moduleauthor:: {{author}} <{{email}}>
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

import typer
from loguru import logger
from rich import print as rprint


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


logger.disable(f"{{module}}")



app = typer.Typer(
    name="{{module}}",
    help="{{description}}",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

verbosity_level = 0

def version_callback(value: bool) -> None:  # noqa FBT001
    """Prints the version of the package."""
    if value:
        rprint(f"[yellow]boardgamegeek[/] version: [bold blue]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def verbosity(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            help="Control output verbosity. Pass this argument multiple times to increase the amount of output.",
            count=True,
        ),
    ] = 0
) -> None:
    verbosity_level = verbose  # noqa: F841


if __name__ == "main":
    app()
