import typer
from loguru import logger

from {{module}} import version, version_callback, verbosity_level

@app.callback(invoke_without_command=True)
@app.command(
    # name="count",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def main(
    debug: Annotated[
        bool,
        typer.Option("--debug")
    ] = False,
    version: Annotated[  # noqa ARG001
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            help="Print version number.",
        ),
    ){
    logger.remove()
    if debug:
        logger.add(
            sys.stderr,
            format="* <red>{elapsed}</red> - <cyan>{module}:{file}:{function}</cyan>:<green>{line}</green> - <yellow>{message}</yellow>",
            colorize=True,
        )
        init_logger(verbose=verbosity_level)
    else:
        logger.add(sys.stderr, format="* <yellow>{message}</yellow>", colorize=True)
        init_logger(verbose=1, msg_format="<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
}