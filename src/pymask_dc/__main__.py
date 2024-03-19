
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

# from skimage.io import imread
from pymask_dc import app, verbosity_level, version, version_callback
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


@app.callback(invoke_without_command=True)
@app.command(
    # name="count",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def generate_mask(
    image_file: Annotated[
        Path,
        typer.Argument(help="Image to generate a mask for")
    ],
    output: Annotated[
        Optional[Path], # noqa: UP007
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
    resolution: Annotated[
        float,
        typer.Option("-r", "--resolution", help="")
    ] = 0.5,
    config_file: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option("--config", help="Path to the configuration file where your Deepcell API token is kept. Only necessary if you haven't setup the api key as per the instructions from Deepcell or you want to just keep it in the config file.")
    ] = None,
    debug: Annotated[
        bool,
        typer.Option("--debug")
    ] = False,
    version: Annotated[
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

    app = Mesmer()
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
    segmentation_predictions = app.predict(im, image_mpp=resolution, compartment=compartment)

    # Uncomment this out when we want to add the GUI where we can display the resulting mask.
    # rgb_images = create_rgb_image(im, channel_colors=["green", "blue"])

    output = Path(f"{image_file.parent}/{image_file.stem}_{compartment}_{resolution}{image_file.stem}") if output is None else output

    match compartment:
        case (CompartmentType.nuclear | CompartmentType.whole_cell):
            output_img = Image.fromarray(segmentation_predictions[0,...,0].astype("float64"))
            output_img.convert("L").save(output)
        case CompartmentType.both:
            output_img = Image.fromarray(segmentation_predictions[0,...].astype("float64"))
            output_img.convert("RGB").save(output)
