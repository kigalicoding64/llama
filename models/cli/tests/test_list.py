# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse
import unittest
from unittest.mock import MagicMock, patch

from llama_models.cli.list import List, _convert_to_model_descriptor, _get_model_size


class TestListHelperFunctions(unittest.TestCase):
    def test_get_model_size_empty_dir(self):
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_rglob.return_value = []
            size = _get_model_size("/fake/path")
            assert size == 0

    def test_get_model_size_with_files(self):
        mock_file1 = MagicMock()
        mock_file1.is_file.return_value = True
        mock_file1.stat.return_value.st_size = 1000

        mock_file2 = MagicMock()
        mock_file2.is_file.return_value = True
        mock_file2.stat.return_value.st_size = 2000

        mock_dir = MagicMock()
        mock_dir.is_file.return_value = False

        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_rglob.return_value = [mock_file1, mock_file2, mock_dir]
            size = _get_model_size("/fake/path")
            assert size == 3000

    @patch("llama_models.cli.list.all_registered_models")
    def test_convert_to_model_descriptor_found(self, mock_models):
        mock_model = MagicMock()
        mock_model.descriptor.return_value = "Llama-3.1:8B"
        mock_models.return_value = [mock_model]

        result = _convert_to_model_descriptor("Llama-3.1-8B")
        assert result == "Llama-3.1:8B"

    @patch("llama_models.cli.list.all_registered_models")
    def test_convert_to_model_descriptor_not_found(self, mock_models):
        mock_models.return_value = []
        result = _convert_to_model_descriptor("unknown-model")
        assert result == "unknown-model"


class TestListCommand(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.list_cmd = List(self.subparsers)

    def test_list_command_parser_created(self):
        assert self.list_cmd.parser is not None

    def test_list_command_has_show_all_argument(self):
        args = self.parser.parse_args(["list", "--show-all"])
        assert args.show_all is True

    def test_list_command_has_downloaded_argument(self):
        args = self.parser.parse_args(["list", "--downloaded"])
        assert args.downloaded is True

    def test_list_command_has_search_argument(self):
        args = self.parser.parse_args(["list", "--search", "llama3"])
        assert args.search == "llama3"

    def test_list_command_short_search_argument(self):
        args = self.parser.parse_args(["list", "-s", "llama4"])
        assert args.search == "llama4"

    @patch("llama_models.cli.list.all_registered_models")
    @patch("llama_models.cli.list.print_table")
    @patch("llama_models.cli.safety_models.prompt_guard_model_skus")
    def test_run_model_list_cmd_filters_featured(self, mock_safety, mock_print, mock_models):
        featured_model = MagicMock()
        featured_model.is_featured = True
        featured_model.descriptor.return_value = "Llama-3.1:8B"
        featured_model.huggingface_repo = "meta-llama/Llama-3.1-8B"
        featured_model.max_seq_length = 131072

        non_featured_model = MagicMock()
        non_featured_model.is_featured = False
        non_featured_model.descriptor.return_value = "Llama-2:7B"

        mock_models.return_value = [featured_model, non_featured_model]
        mock_safety.return_value = []

        args = self.parser.parse_args(["list"])
        args.func(args)

        mock_print.assert_called_once()
        rows = mock_print.call_args[0][0]
        assert len(rows) == 1
        assert rows[0][0] == "Llama-3.1:8B"

    @patch("llama_models.cli.list.all_registered_models")
    @patch("llama_models.cli.list.print_table")
    @patch("llama_models.cli.safety_models.prompt_guard_model_skus")
    def test_run_model_list_cmd_show_all(self, mock_safety, mock_print, mock_models):
        featured_model = MagicMock()
        featured_model.is_featured = True
        featured_model.descriptor.return_value = "Llama-3.1:8B"
        featured_model.huggingface_repo = "meta-llama/Llama-3.1-8B"
        featured_model.max_seq_length = 131072

        non_featured_model = MagicMock()
        non_featured_model.is_featured = False
        non_featured_model.descriptor.return_value = "Llama-2:7B"
        non_featured_model.huggingface_repo = "meta-llama/Llama-2-7b"
        non_featured_model.max_seq_length = 4096

        mock_models.return_value = [featured_model, non_featured_model]
        mock_safety.return_value = []

        args = self.parser.parse_args(["list", "--show-all"])
        args.func(args)

        mock_print.assert_called_once()
        rows = mock_print.call_args[0][0]
        assert len(rows) == 2

    @patch("llama_models.cli.list.all_registered_models")
    @patch("llama_models.cli.list.print_table")
    @patch("llama_models.cli.safety_models.prompt_guard_model_skus")
    def test_run_model_list_cmd_search_filter(self, mock_safety, mock_print, mock_models):
        model1 = MagicMock()
        model1.is_featured = True
        model1.descriptor.return_value = "Llama-3.1:8B"
        model1.huggingface_repo = "meta-llama/Llama-3.1-8B"
        model1.max_seq_length = 131072

        model2 = MagicMock()
        model2.is_featured = True
        model2.descriptor.return_value = "Llama-4-Scout:17B"
        model2.huggingface_repo = "meta-llama/Llama-4-Scout"
        model2.max_seq_length = 10485760

        mock_models.return_value = [model1, model2]
        mock_safety.return_value = []

        args = self.parser.parse_args(["list", "--search", "llama-4"])
        args.func(args)

        mock_print.assert_called_once()
        rows = mock_print.call_args[0][0]
        assert len(rows) == 1
        assert rows[0][0] == "Llama-4-Scout:17B"

    @patch("llama_models.cli.list.all_registered_models")
    @patch("builtins.print")
    @patch("llama_models.cli.safety_models.prompt_guard_model_skus")
    def test_run_model_list_cmd_search_no_results(self, mock_safety, mock_print, mock_models):
        mock_models.return_value = []
        mock_safety.return_value = []

        args = self.parser.parse_args(["list", "--search", "nonexistent"])
        args.func(args)

        mock_print.assert_called_with("Did not find any model matching `nonexistent`.")


if __name__ == "__main__":
    unittest.main()
