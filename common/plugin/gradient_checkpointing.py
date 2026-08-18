"""Gradient checkpointing runtime plugins.

Ecosystems expose gradient checkpointing through different entry-point APIs --
transformers' ``gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)``,
diffusers' no-argument ``enable_gradient_checkpointing()``, and the
``set_gradient_checkpointing(**kwargs)`` that vendored model code in this repo
implements natively -- so each gets its own plugin class rather than one
branching on library.

Like every runtime plugin this also runs on the EMA by default; checkpointing
is inert on the eval-mode EMA, so that is harmless (``ema: {enabled: false}``
opts out).

Example::

    runtime:
      plugins:
        - module: common.plugin.gradient_checkpointing
          class_name: TransformersGradientCheckpointing
          use_reentrant: false
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ..model.runtime import RuntimePlugin


class TransformersGradientCheckpointing(RuntimePlugin):
    """Enable transformers-style gradient checkpointing (after the core runtime)."""

    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        kwargs = {k: v for k, v in self.config.items() if k not in ("module", "class_name")}
        state["model"].gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs or None)
        return state


class DiffusersGradientCheckpointing(RuntimePlugin):
    """Enable diffusers-style gradient checkpointing (after the core runtime)."""

    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        state["model"].enable_gradient_checkpointing()
        return state


class NativeGradientCheckpointing(RuntimePlugin):
    """Call a model's own ``set_gradient_checkpointing(**kwargs)``.

    The third entry-point API, and the one the vendored model code in this repo
    implements itself, so it takes whatever keyword arguments that model chose
    to expose (``enable``, ``gc_start_idx``, ``gc_step``, ...) straight from the
    plugin config. Nothing here is specific to a model family -- the only
    requirement is that the method exists.
    """

    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if not isinstance(model, nn.Module):
            raise TypeError(
                "NativeGradientCheckpointing requires state['model'] to be a torch.nn.Module, "
                f"got {type(model).__name__}"
            )

        set_gradient_checkpointing = getattr(model, "set_gradient_checkpointing", None)
        if not callable(set_gradient_checkpointing):
            raise TypeError(
                "NativeGradientCheckpointing requires the model to implement callable "
                "set_gradient_checkpointing(...)"
            )

        kwargs = {k: v for k, v in self.config.items() if k not in ("module", "class_name")}
        set_gradient_checkpointing(**kwargs)
        return state
