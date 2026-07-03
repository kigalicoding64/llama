# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Vercel entrypoint for the online Llama API."""

from models.online.server import app

__all__ = ["app"]
