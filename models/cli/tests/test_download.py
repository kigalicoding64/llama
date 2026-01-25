# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from llama_models.cli.download import (
    CustomTransferSpeedColumn,
    Download,
    DownloadError,
    DownloadTask,
    ModelEntry,
    ParallelDownloader,
    setup_download_parser,
)


class TestDownloadTask(unittest.TestCase):
    def test_download_task_creation(self):
        task = DownloadTask(url="https://example.com/model.bin", output_file="/tmp/model.bin")
        assert task.url == "https://example.com/model.bin"
        assert task.output_file == "/tmp/model.bin"
        assert task.total_size == 0
        assert task.downloaded_size == 0
        assert task.task_id is None
        assert task.retries == 0
        assert task.max_retries == 3

    def test_download_task_with_all_fields(self):
        task = DownloadTask(
            url="https://example.com/model.bin",
            output_file="/tmp/model.bin",
            total_size=1000000,
            downloaded_size=500000,
            task_id=1,
            retries=2,
            max_retries=5,
        )
        assert task.total_size == 1000000
        assert task.downloaded_size == 500000
        assert task.task_id == 1
        assert task.retries == 2
        assert task.max_retries == 5


class TestDownloadError(unittest.TestCase):
    def test_download_error_is_exception(self):
        assert issubclass(DownloadError, Exception)

    def test_download_error_can_be_raised(self):
        with self.assertRaises(DownloadError):
            raise DownloadError("Test error")


class TestModelEntry(unittest.TestCase):
    def test_model_entry_creation(self):
        entry = ModelEntry(model_id="Llama-3.1:8B", files={"model.bin": "https://example.com/model.bin"})
        assert entry.model_id == "Llama-3.1:8B"
        assert entry.files == {"model.bin": "https://example.com/model.bin"}


class TestCustomTransferSpeedColumn(unittest.TestCase):
    def test_render_with_speed(self):
        column = CustomTransferSpeedColumn()
        mock_task = MagicMock()
        mock_task.finished = False
        mock_task.speed = 1024 * 1024

        result = column.render(mock_task)
        assert result is not None

    def test_render_finished_task(self):
        column = CustomTransferSpeedColumn()
        mock_task = MagicMock()
        mock_task.finished = True

        result = column.render(mock_task)
        assert result is not None


class TestDownloadCommand(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.download_cmd = Download(self.subparsers)

    def test_download_command_parser_created(self):
        assert self.download_cmd.parser is not None


class TestSetupDownloadParser(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        setup_download_parser(self.parser)

    def test_parser_has_model_id_argument(self):
        args = self.parser.parse_args(["--model-id", "Llama-3.1:8B"])
        assert args.model_id == "Llama-3.1:8B"

    def test_parser_has_source_argument(self):
        args = self.parser.parse_args(["--source", "meta"])
        assert args.source == "meta"

    def test_parser_source_choices(self):
        args = self.parser.parse_args(["--source", "huggingface"])
        assert args.source == "huggingface"

        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--source", "invalid"])

    def test_parser_has_hf_token_argument(self):
        args = self.parser.parse_args(["--hf-token", "hf_xxx"])
        assert args.hf_token == "hf_xxx"

    def test_parser_has_ignore_patterns_argument(self):
        args = self.parser.parse_args(["--ignore-patterns", "*.md"])
        assert args.ignore_patterns == "*.md"

    def test_parser_has_max_parallel_argument(self):
        args = self.parser.parse_args(["--max-parallel", "5"])
        assert args.max_parallel == 5

    def test_parser_has_meta_url_argument(self):
        args = self.parser.parse_args(["--meta-url", "https://example.com"])
        assert args.meta_url == "https://example.com"

    def test_parser_has_manifest_file_argument(self):
        args = self.parser.parse_args(["--manifest-file", "/path/to/manifest.json"])
        assert args.manifest_file == "/path/to/manifest.json"


class TestParallelDownloader(unittest.TestCase):
    def test_parallel_downloader_initialization(self):
        downloader = ParallelDownloader()
        assert downloader.max_concurrent_downloads == 3
        assert downloader.buffer_size == 1024 * 1024
        assert downloader.timeout == 30

    def test_parallel_downloader_custom_params(self):
        downloader = ParallelDownloader(max_concurrent_downloads=5, buffer_size=2048, timeout=60)
        assert downloader.max_concurrent_downloads == 5
        assert downloader.buffer_size == 2048
        assert downloader.timeout == 60

    def test_verify_file_integrity_with_matching_size(self):
        downloader = ParallelDownloader()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            temp_path = f.name

        try:
            file_size = Path(temp_path).stat().st_size
            task = DownloadTask(url="https://example.com/test.bin", output_file=temp_path, total_size=file_size)
            result = downloader.verify_file_integrity(task)
            assert result is True
        finally:
            Path(temp_path).unlink()

    def test_verify_file_integrity_file_not_found(self):
        downloader = ParallelDownloader()
        task = DownloadTask(url="https://example.com/test.bin", output_file="/nonexistent/path", total_size=100)
        result = downloader.verify_file_integrity(task)
        assert result is False

    def test_has_disk_space_sufficient(self):
        downloader = ParallelDownloader()

        task = DownloadTask(
            url="https://example.com/small.bin",
            output_file="/tmp/small.bin",
            total_size=1024,
        )

        result = downloader.has_disk_space([task])
        assert result is True

    def test_prepare_download_creates_directory(self):
        downloader = ParallelDownloader()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "model.bin"
            task = DownloadTask(url="https://example.com/model.bin", output_file=str(output_path))

            asyncio.run(downloader.prepare_download(task))
            assert output_path.parent.exists()


class TestRetryWithExponentialBackoff(unittest.TestCase):
    def test_retry_success_on_first_attempt(self):
        downloader = ParallelDownloader()
        task = DownloadTask(url="https://example.com", output_file="/tmp/test.bin")

        async def mock_func():
            return "success"

        result = asyncio.run(downloader.retry_with_exponential_backoff(task, mock_func))
        assert result == "success"

    def test_retry_fails_eventually(self):
        downloader = ParallelDownloader()
        task = DownloadTask(url="https://example.com", output_file="/tmp/test.bin", max_retries=2)

        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise DownloadError("Network error")

        with self.assertRaises(DownloadError):
            asyncio.run(downloader.retry_with_exponential_backoff(task, failing_func))

        assert call_count == task.max_retries


if __name__ == "__main__":
    unittest.main()
