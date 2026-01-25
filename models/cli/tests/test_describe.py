# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse
import unittest
from unittest.mock import MagicMock, patch

from llama_models.cli.describe import Describe


class TestDescribeCommand(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.describe_cmd = Describe(self.subparsers)

    def test_describe_command_parser_created(self):
        assert self.describe_cmd.parser is not None

    def test_describe_command_requires_model_id(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["describe"])

    def test_describe_command_accepts_model_id(self):
        args = self.parser.parse_args(["describe", "--model-id", "Llama-3.1:8B"])
        assert args.model_id == "Llama-3.1:8B"

    def test_describe_command_short_model_id_argument(self):
        args = self.parser.parse_args(["describe", "-m", "Llama-4-Scout:17B"])
        assert args.model_id == "Llama-4-Scout:17B"

    @patch("llama_models.cli.describe.resolve_model")
    @patch("llama_models.cli.describe.print_table")
    @patch("llama_models.cli.safety_models.prompt_guard_model_sku_map")
    def test_run_describe_cmd_success(self, mock_safety_map, mock_print, mock_resolve):
        mock_safety_map.return_value = {}

        mock_model = MagicMock()
        mock_model.descriptor.return_value = "Llama-3.1:8B"
        mock_model.huggingface_repo = "meta-llama/Llama-3.1-8B"
        mock_model.description = "A powerful language model"
        mock_model.max_seq_length = 131072
        mock_model.quantization_format.value = "bf16"
        mock_model.arch_args = {"dim": 4096, "n_layers": 32}

        mock_resolve.return_value = mock_model

        args = self.parser.parse_args(["describe", "-m", "Llama-3.1:8B"])
        args.func(args)

        mock_print.assert_called_once()
        rows = mock_print.call_args[0][0]
        headers = mock_print.call_args[0][1]

        assert headers[0] == "Model"
        assert headers[1] == "Llama-3.1:8B"

        row_dict = {row[0]: row[1] for row in rows}
        assert row_dict["Hugging Face ID"] == "meta-llama/Llama-3.1-8B"
        assert row_dict["Description"] == "A powerful language model"
        assert row_dict["Context Length"] == "128K tokens"
        assert row_dict["Weights format"] == "bf16"

    @patch("llama_models.cli.describe.resolve_model")
    @patch("llama_models.cli.safety_models.prompt_guard_model_sku_map")
    def test_run_describe_cmd_model_not_found(self, mock_safety_map, mock_resolve):
        mock_safety_map.return_value = {}
        mock_resolve.return_value = None

        args = self.parser.parse_args(["describe", "-m", "NonExistent:Model"])

        with self.assertRaises(SystemExit):
            args.func(args)

    @patch("llama_models.cli.describe.resolve_model")
    @patch("llama_models.cli.describe.print_table")
    @patch("llama_models.cli.safety_models.prompt_guard_model_sku_map")
    def test_run_describe_cmd_no_huggingface_repo(self, mock_safety_map, mock_print, mock_resolve):
        mock_safety_map.return_value = {}

        mock_model = MagicMock()
        mock_model.descriptor.return_value = "Custom:Model"
        mock_model.huggingface_repo = None
        mock_model.description = "Custom model"
        mock_model.max_seq_length = 4096
        mock_model.quantization_format.value = "fp16"
        mock_model.arch_args = {}

        mock_resolve.return_value = mock_model

        args = self.parser.parse_args(["describe", "-m", "Custom:Model"])
        args.func(args)

        mock_print.assert_called_once()
        rows = mock_print.call_args[0][0]
        row_dict = {row[0]: row[1] for row in rows}
        assert row_dict["Hugging Face ID"] == "<Not Available>"

    @patch("llama_models.cli.describe.resolve_model")
    @patch("llama_models.cli.describe.print_table")
    @patch("llama_models.cli.safety_models.prompt_guard_model_sku_map")
    def test_run_describe_cmd_prompt_guard_model(self, mock_safety_map, mock_print, mock_resolve):
        mock_model = MagicMock()
        mock_model.descriptor.return_value = "Prompt-Guard:86M"
        mock_model.huggingface_repo = "meta-llama/Prompt-Guard-86M"
        mock_model.description = "Safety model"
        mock_model.max_seq_length = 512
        mock_model.quantization_format.value = "bf16"
        mock_model.arch_args = {}

        mock_safety_map.return_value = {"Prompt-Guard:86M": mock_model}
        mock_resolve.return_value = None

        args = self.parser.parse_args(["describe", "-m", "Prompt-Guard:86M"])
        args.func(args)

        mock_print.assert_called_once()
        mock_resolve.assert_not_called()


class TestDescribeCommandContextLengthFormatting(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.describe_cmd = Describe(self.subparsers)

    @patch("llama_models.cli.describe.resolve_model")
    @patch("llama_models.cli.describe.print_table")
    @patch("llama_models.cli.safety_models.prompt_guard_model_sku_map")
    def test_context_length_formatting_128k(self, mock_safety_map, mock_print, mock_resolve):
        mock_safety_map.return_value = {}
        mock_model = MagicMock()
        mock_model.descriptor.return_value = "Test:Model"
        mock_model.huggingface_repo = "test/model"
        mock_model.description = "Test"
        mock_model.max_seq_length = 131072
        mock_model.quantization_format.value = "bf16"
        mock_model.arch_args = {}
        mock_resolve.return_value = mock_model

        args = self.parser.parse_args(["describe", "-m", "Test:Model"])
        args.func(args)

        rows = mock_print.call_args[0][0]
        row_dict = {row[0]: row[1] for row in rows}
        assert row_dict["Context Length"] == "128K tokens"

    @patch("llama_models.cli.describe.resolve_model")
    @patch("llama_models.cli.describe.print_table")
    @patch("llama_models.cli.safety_models.prompt_guard_model_sku_map")
    def test_context_length_formatting_10m(self, mock_safety_map, mock_print, mock_resolve):
        mock_safety_map.return_value = {}
        mock_model = MagicMock()
        mock_model.descriptor.return_value = "Llama-4-Scout:17B"
        mock_model.huggingface_repo = "meta-llama/Llama-4-Scout"
        mock_model.description = "Llama 4 Scout MoE"
        mock_model.max_seq_length = 10485760
        mock_model.quantization_format.value = "bf16"
        mock_model.arch_args = {}
        mock_resolve.return_value = mock_model

        args = self.parser.parse_args(["describe", "-m", "Llama-4-Scout:17B"])
        args.func(args)

        rows = mock_print.call_args[0][0]
        row_dict = {row[0]: row[1] for row in rows}
        assert row_dict["Context Length"] == "10240K tokens"


if __name__ == "__main__":
    unittest.main()
