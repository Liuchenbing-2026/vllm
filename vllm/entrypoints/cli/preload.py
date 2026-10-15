# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.model_executor.model_loader.weight_cache.daemon import add_cli_args, run
from vllm.utils.argparse_utils import FlexibleArgumentParser


class PreloadSubcommand(CLISubcommand):
    """Keep model weights resident for engines using ``--load-format ipc_cache``."""

    name = "preload"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        run(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Preload model weights for fast engine restarts.",
            description="Launch persistent weight cache daemons (one per TP rank).",
            usage="vllm preload [model] [options]",
        )
        parser.add_argument("model_tag", nargs="?", help="Model name or local path.")
        return add_cli_args(parser)


def cmd_init() -> list[CLISubcommand]:
    return [PreloadSubcommand()]
