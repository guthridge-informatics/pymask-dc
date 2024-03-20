
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import tomli
import typer
from deepcell.applications import Mesmer
from loguru import logger

# from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay
from PIL import Image
from rich import print as rprint

# from skimage.io import imread
from pymask_dc import version
from pymask_dc.logging import init_logger


class RGB(int, Enum):
    red = 0
    green = 1
    blue = 2

class BGR(int, Enum):
    blue = 0
    green = 1
    red = 2

class ColorOrder(str, Enum):
    RGB = "RGB"
    BGR = "BGR"

class CompartmentType(str, Enum):
    nuclear = "nuclear"
    whole_cell = "whole-cell"
    both = "both"

logger.disable("pymask_dc")

verbosity_level = 0

app = typer.Typer(
    name="pymask_dc",
    help="Commandline interface for generating masks using Deepcell Mesmer",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

def version_callback(value: bool) -> None:  # FBT001
    """Prints the version of the package."""
    if value:
        rprint(f"[yellow]pymask_dc[/] version: [bold blue]{version}[/]")
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

@app.callback(invoke_without_command=True)
@app.command(
    name="generate_mask",
    no_args_is_help=True,
    #context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def generate_mask(
    image_file: Annotated[
        Path,
        typer.Argument(help="Image to generate a mask for")
    ],
    output: Annotated[
        Optional[Path], # noqa: UP007
        list[Path],
        typer.Option("-o", "--output", help="Name to give output file. If none is given, the results will be named using the input filename, resolution, and compartment mask.")
    ] = None,
    mode: Annotated[
        ColorOrder,
        typer.Option("-l", "--colororder", help="Specify the order of the color channels in the source image.  Most likely, you want to leave this one alone.")
    ] = ColorOrder.RGB,
    compartment: Annotated[
        CompartmentType,
        typer.Option("-c", "--compartment")
    ] = CompartmentType.both,
    separate: Annotated[  # noqa: FBT002
        bool,
        typer.Option("-s", "--separate", help="If generating masks for both the nucleus and whole-cell (i.e. passing 'both' for 'compartment') should the resul be output as a combined image or separate files?")
    ] = False,
    resolution: Annotated[
        float,
        typer.Option("-r", "--resolution", help="")
    ] = 0.5,
    config_file: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option("--config", help="Path to the configuration file where your Deepcell API token is kept. Only necessary if you haven't setup the api key as per the instructions from Deepcell or you want to just keep it in the config file.")
    ] = None,
    debug: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--debug")
    ] = False,
    version: Annotated[  # noqa: ARG001, ARG001
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            help="Print version number.",
        ),
    ] = False
    ) -> None:

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

    if "DEEPCELL_ACCESS_TOKEN" not in os.environ:
        if config_file is None:
            config_file = Path().home().joinpath(".config", "pymask.toml")
        elif not isinstance(config_file, Path):
            config_file = Path(config_file)

        if config_file.exists():
            with config_file.open("rb") as cf:
                config = tomli.load(cf)
        else:
            msg = f"The config file is not found at {config_file}"
            raise FileNotFoundError(msg)

        os.environ["DEEPCELL_ACCESS_TOKEN"] = config["API"]["KEY"]

    if image_file.exists():
        img = np.array(Image.open(image_file)).astype("float64")
    else:
        msg = f"Cannot find file at {image_file}"
        raise FileNotFoundError(msg)

    match mode:
        case ColorOrder.RGB:
            im = np.stack((img[:,:,2], img[:,:,1]), axis=-1)
        case ColorOrder.BGR:
            im = np.stack((img[:,:,0], img[:,:,1]), axis=-1)

    im = np.expand_dims(im, 0)
    app = Mesmer()
    segmentation_result = app.predict(im, image_mpp=resolution, compartment=compartment)

    # Uncomment this out when we want to add the GUI where we can display the resulting mask.
    # rgb_images = create_rgb_image(im, channel_colors=["green", "blue"])

    output = Path(f"{image_file.parent}/{image_file.stem}_{compartment}_{resolution}{image_file.suffix}") if output is None else output
    logger.info(f"{output=}")
    match compartment:
        case (CompartmentType.nuclear | CompartmentType.whole_cell):
            output_img = Image.fromarray(segmentation_result[0,...,0].astype("float64"))
            output_img.convert("L").save(output)
        case CompartmentType.both:
            if separate:
                if isinstance(output, Path):
                    output_name_list = [
                        Path(f"{output.parent}/{output.stem}_nuclear_{image_file.stem}"),
                        Path(f"{output.parent}/{output.stem}_whole-cell_{image_file.stem}")
                    ]
                else:
                    output_name_list = output

                nuclear_img = Image.fromarray(segmentation_result[0,...,1].astype("float64"))
                nuclear_img.convert("L").save(output_name_list[0])

                wc_img = Image.fromarray(segmentation_result[0,...,0].astype("float64"))
                wc_img.convert("L").save(output_name_list[1])
            else:
                fake_rgb = np.multiply(
                    np.dstack(
                        (
                            np.zeros((*segmentation_result[0,...,0].shape,1)), #fake the red channel
                            segmentation_result[0,...,0],
                            segmentation_result[0,...,1],
                        )
                    ),
                    255.999
                ).astype(np.uint8)
                output_img = Image.fromarray(fake_rgb)
                logger.info(f"about to write a file to {output=}")
                output_img.convert("RGB").save(output)

if __name__ == "main":
    app()
