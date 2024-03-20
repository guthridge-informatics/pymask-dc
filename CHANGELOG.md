# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [0.3.0] - 2024-03-20

## Changed

- Change writing single channel masks from 64-bit to 32-bit floating point tiff images
- Stop writing the log file by default

# [0.2.0] - 2024-03-20

## Added

- Option to write either separate mask files or a combined mask file when passing the "both" option for "compartment"

## Fixed

- Should no longer crash when attempting to save the mask generated using the "both" option for "compartment"

# [0.1.0] - 2024-03-19

## Added

- Everything

[0.3.0]: https://github.com/milescsmith/pymask_dc/compare/0.1.0...0.3.0
[0.2.0]: https://github.com/milescsmith/pymask_dc/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/milescsmith/pymask_dc/releases/tag/0.1.0