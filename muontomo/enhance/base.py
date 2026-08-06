"""Technique interface + registry. Adding an enhancer = one module that defines
an Enhancer and calls register(); nothing in the core pipeline changes."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .context import EnhanceContext


@runtime_checkable
class Enhancer(Protocol):
    name: str

    def enhance(self, ctx: EnhanceContext) -> np.ndarray:
        """Return the enhanced ceiling layer as a 2D array on ctx's (xs, ys) grid."""
        ...


REGISTRY: dict[str, Enhancer] = {}


def register(enhancer: Enhancer) -> Enhancer:
    REGISTRY[enhancer.name] = enhancer
    return enhancer


class FnEnhancer:
    """Wrap a plain function as an Enhancer (keeps technique modules terse)."""

    def __init__(self, name: str, fn: Callable[[EnhanceContext], np.ndarray]):
        self.name = name
        self._fn = fn

    def enhance(self, ctx: EnhanceContext) -> np.ndarray:
        return self._fn(ctx)
