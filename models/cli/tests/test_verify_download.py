# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_models.cli.verify_download import (
    VerificationResult,
    VerifyDownload,
    calculate_sha256,
    load_checksums,
    setup_verify_download_parser,
    verify_files,
)


class TestVerificationResult(unittest.TestCase):
    def test_verification_result_creation(self):
        result = VerificationResult(
            filename="model.safetensors",
            expected_hash="abc123",
            actual_hash="abc123",
            exists=True,
            matches=True,
        )
        assert result.filename == "model.safetensors"
        assert result.expected_hash == "abc123"
        assert result.actual_hash == "abc123"
        assert result.exists is True
        assert result.matches is True

    def test_verification_result_mismatch(self):
        result = VerificationResult(
            filename="model.safetensors",
            expected_hash="abc123",
            actual_hash="def456",
            exists=True,
            matches=False,
        )
        assert result.matches is False

    def test_verification_result_missing_file(self):
        result = VerificationResult(
            filename="missing.bin",
            expected_hash="abc123",
            actual_hash=None,
            exists=False,
            matches=False,
        )
        assert result.exists is False
        assert result.actual_hash is None


class TestCalculateSha256(unittest.TestCase):
    def test_calculate_sha256_small_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Hello, World!")
            f.flush()
            temp_path = Path(f.name)

        try:
            expected_hash = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
            actual_hash = calculate_sha256(temp_path)
            assert actual_hash == expected_hash
        finally:
            temp_path.unlink()

    def test_calculate_sha256_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            actual_hash = calculate_sha256(temp_path)
            assert actual_hash == expected_hash
        finally:
            temp_path.unlink()


class TestLoadChecksums(unittest.TestCase):
    def test_load_checksums_standard_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".chk", delete=False) as f:
            f.write("abc123def456  model.safetensors\n")
            f.write("789xyz000111  tokenizer.model\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            checksums = load_checksums(temp_path)
            assert len(checksums) == 2
            assert checksums["model.safetensors"] == "abc123def456"
            assert checksums["tokenizer.model"] == "789xyz000111"
        finally:
            temp_path.unlink()

    def test_load_checksums_with_leading_dotslash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".chk", delete=False) as f:
            f.write("abc123  ./model.safetensors\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            checksums = load_checksums(temp_path)
            assert checksums["model.safetensors"] == "abc123"
        finally:
            temp_path.unlink()

    def test_load_checksums_with_subdirectory(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".chk", delete=False) as f:
            f.write("abc123  subdir/model.safetensors\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            checksums = load_checksums(temp_path)
            assert checksums["subdir/model.safetensors"] == "abc123"
        finally:
            temp_path.unlink()

    def test_load_checksums_empty_lines_ignored(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".chk", delete=False) as f:
            f.write("abc123  model.safetensors\n")
            f.write("\n")
            f.write("def456  tokenizer.model\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            checksums = load_checksums(temp_path)
            assert len(checksums) == 2
        finally:
            temp_path.unlink()


class TestVerifyDownloadCommand(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.verify_cmd = VerifyDownload(self.subparsers)

    def test_verify_download_command_parser_created(self):
        assert self.verify_cmd.parser is not None

    def test_verify_download_command_requires_model_id(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["verify-download"])

    def test_verify_download_command_accepts_model_id(self):
        args = self.parser.parse_args(["verify-download", "--model-id", "Llama-3.1:8B"])
        assert args.model_id == "Llama-3.1:8B"


class TestSetupVerifyDownloadParser(unittest.TestCase):
    def test_setup_adds_model_id_argument(self):
        parser = argparse.ArgumentParser()
        setup_verify_download_parser(parser)

        args = parser.parse_args(["--model-id", "test-model"])
        assert args.model_id == "test-model"

    def test_setup_sets_func_default(self):
        parser = argparse.ArgumentParser()
        setup_verify_download_parser(parser)

        args = parser.parse_args(["--model-id", "test"])
        assert hasattr(args, "func")
        assert callable(args.func)


class TestVerifyFilesFunction(unittest.TestCase):
    def test_verify_files_all_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            file1 = model_dir / "model.bin"
            file1.write_bytes(b"test content 1")
            file1_hash = calculate_sha256(file1)

            file2 = model_dir / "tokenizer.model"
            file2.write_bytes(b"test content 2")
            file2_hash = calculate_sha256(file2)

            checksums = {"model.bin": file1_hash, "tokenizer.model": file2_hash}

            mock_console = MagicMock()
            results = verify_files(model_dir, checksums, mock_console)

            assert len(results) == 2
            for result in results:
                assert result.exists is True
                assert result.matches is True

    def test_verify_files_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            checksums = {"nonexistent.bin": "abc123"}

            mock_console = MagicMock()
            results = verify_files(model_dir, checksums, mock_console)

            assert len(results) == 1
            assert results[0].exists is False
            assert results[0].matches is False

    def test_verify_files_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            file1 = model_dir / "model.bin"
            file1.write_bytes(b"actual content")

            checksums = {"model.bin": "wrong_hash_value"}

            mock_console = MagicMock()
            results = verify_files(model_dir, checksums, mock_console)

            assert len(results) == 1
            assert results[0].exists is True
            assert results[0].matches is False
            assert results[0].actual_hash != "wrong_hash_value"


class TestRunVerifyCommand(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        setup_verify_download_parser(self.parser)

    @patch("llama_models.utils.model_utils.model_local_dir")
    def test_run_verify_cmd_model_dir_not_found(self, mock_local_dir):
        mock_local_dir.return_value = "/nonexistent/path"

        args = self.parser.parse_args(["--model-id", "test-model"])

        with self.assertRaises(SystemExit):
            args.func(args)

    @patch("llama_models.utils.model_utils.model_local_dir")
    def test_run_verify_cmd_checklist_not_found(self, mock_local_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_local_dir.return_value = tmpdir

            args = self.parser.parse_args(["--model-id", "test-model"])

            with self.assertRaises(SystemExit):
                args.func(args)


if __name__ == "__main__":
    unittest.main()
