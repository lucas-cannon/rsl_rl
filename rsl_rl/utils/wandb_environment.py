# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from collections.abc import Mapping


def resolve_wandb_entity(environment: Mapping[str, str] | None = None) -> str | None:
    """Return the configured W&B namespace, accepting the legacy variable."""
    values = os.environ if environment is None else environment
    return values.get("WANDB_ENTITY") or values.get("WANDB_USERNAME") or None
