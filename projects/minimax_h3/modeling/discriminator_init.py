# SPDX-License-Identifier: Apache-2.0
"""Placement-time initialization for the MiniMax H3 DMD2 discriminator.

The discriminator is new weight -- no checkpoint carries it -- so its meta shell
has to be materialized and initialized by hand. This runs ``before_placement``:
the parameters are still whole CPU tensors there, so ``xavier_uniform_`` sees the
real fan-in/fan-out and the core placement op shards already-initialized values.

Example::

    placement:
      fsdp: { wrap_modules: [MiniMaxH3DiscriminatorBlock] }
      plugins:
        - module: projects.minimax_h3.modeling.discriminator_init
          class_name: InitMiniMaxH3Discriminator
          seed: 1019
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from common.model.placement import PlacementPlugin

from .discriminator import MiniMaxH3DMD2Discriminator


class InitMiniMaxH3Discriminator(PlacementPlugin):
    """Materialize and deterministically initialize meta discriminators."""

    def before_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if isinstance(model, nn.Module) and hasattr(model, "discriminator"):
            model = model.discriminator
        if not isinstance(model, nn.Module):
            return state

        generator = torch.Generator(device="cpu").manual_seed(int(self.config["seed"]))

        for module in model.modules():
            if not isinstance(module, MiniMaxH3DMD2Discriminator):
                continue
            if any(parameter.is_meta for parameter in module.parameters()):
                module.to_empty(device=torch.device("cpu"))
                module.reset_parameters(generator=generator)

        return state


__all__ = ["InitMiniMaxH3Discriminator"]
