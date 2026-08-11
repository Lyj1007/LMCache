# SPDX-License-Identifier: Apache-2.0
"""Device detection helpers, decoupled from ``lmcache.v1.platform``.

Kept in its own module (rather than in ``platform/__init__.py``) so that
peers such as :mod:`lmcache.v1.platform.torch_ops` can import the
detection primitives at the top of the file without introducing an
import cycle -- ``platform/__init__.py`` itself pulls in
``base_device_ops``, which in turn pulls in ``torch_ops``, so any name
that ``torch_ops`` needs from the platform package must live *outside*
that init chain.

The registry of :class:`DeviceSpec` subclasses and the detected torch
device module are both built lazily on first access and then cached
process-wide.
"""

# Standard
from functools import lru_cache
from typing import TYPE_CHECKING, Any
import os

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Kunlun KLX_XPU detection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def is_kunlun_klx_xpu() -> bool:
    """Detect whether the current environment is Kunlun KLX_XPU with xmlir.

    On Kunlun KLX_XPU, ``torch_xmlir`` is loaded and provides CUDA API
    compatibility, so ``torch.cuda.is_available()`` reports True even though
    the underlying hardware is Kunlun.  The KLX_XPU offload path uses this flag to
    win device auto-detection ahead of the generic CUDA spec and to pick the
    engine-driven transfer context.

    Kept in this low-level module (rather than ``lmcache.__init__``) so that
    :meth:`DeviceSpec.is_available` implementations and the multiprocess
    transfer router can share it without importing ``lmcache.__init__`` and
    triggering the platform import cycle.

    Returns:
        True when ``torch_xmlir`` can be imported, False otherwise.
    """
    try:
        # Third Party
        import torch_xmlir  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_device_registry() -> "dict[str, DeviceSpec]":
    """Discover and instantiate every :class:`DeviceSpec` subclass.

    Discovery is deferred until first use so importing this module
    stays cheap and side-effect free.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec
    from lmcache.v1.utils.subclass_discovery import discover_subclasses

    return {
        spec.device_type: spec
        for spec in [
            cls()
            for cls in discover_subclasses(
                "lmcache.v1.platform",
                DeviceSpec,  # type: ignore[type-abstract]
                module_filter=lambda name: not name.startswith(("_", "base")),
                require_defined_in_module=True,
                on_import_error=lambda name, exc: None,
            )
        ]
    }


def _detect_device() -> tuple[Any, str]:
    """Detect the available accelerator via the device registry.

    Returns:
        tuple[Any, str]: A tuple of (torch_device_module, device_type_string).
            When torch is not installed (CLI-only mode), returns
            ``(None, "cpu")``.
    """
    try:
        # Third Party
        import torch
    except ImportError as e:
        logger.warning("load torch failed, error is %s", e)
        return None, "cpu"  # fallback for CLI-only environments

    registry = _build_device_registry()

    # Check DEVICE_TYPE environment variable for forced device selection.
    env_device_type = os.environ.get("DEVICE_TYPE")
    if env_device_type is not None:
        env_device_type = env_device_type.strip().lower()
        spec = registry.get(env_device_type)
        if spec is not None and spec.is_available():
            torch_module = getattr(torch, spec.torch_module_name, None)
            if torch_module is not None:
                return torch_module, spec.device_type
            else:
                logger.warning(
                    "DEVICE_TYPE=%r is available but torch module [%s] not found, "
                    "falling back to auto-detection.",
                    env_device_type,
                    spec.torch_module_name,
                )
        else:
            logger.warning(
                "DEVICE_TYPE=%r is not available or not registered, "
                "falling back to auto-detection.",
                env_device_type,
            )

    # Kunlun KLX_XPU exposes torch.cuda via xmlir, so CudaDeviceSpec.is_available()
    # would otherwise win the generic (alphabetical) scan below and bind the
    # CUDA native .so that cannot run on Kunlun hardware. Give the KLX_XPU spec
    # priority so KlxXpuDeviceOps (THP hugepages + xmlir memcpy) is selected.
    if is_kunlun_klx_xpu():
        klx_xpu_spec = registry.get("klx_xpu")
        if klx_xpu_spec is not None and klx_xpu_spec.is_available():
            torch_module = getattr(torch, klx_xpu_spec.torch_module_name, None)
            if torch_module is not None:
                return torch_module, klx_xpu_spec.device_type

    for spec in registry.values():
        if not spec.is_available():
            continue

        torch_module = getattr(torch, spec.torch_module_name, None)
        if torch_module is not None:
            return torch_module, spec.device_type
        else:
            logger.warning(
                "device [%s] is available, but torch module [%s] is not found.",
                spec.device_type,
                spec.torch_module_name,
            )

    # No accelerator found -- fall back to CPU stub
    # First Party
    from lmcache.v1.platform.cpu.stub_cpu_device import StubCPUDevice

    return StubCPUDevice("cpu"), "cpu"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_device_spec(device_type: str) -> "DeviceSpec | None":
    """Return the :class:`DeviceSpec` registered for *device_type*, if any."""
    return _build_device_registry().get(device_type)


@lru_cache(maxsize=1)
def get_torch_device() -> tuple[Any, str]:
    """Return the cached ``(torch_dev, torch_device_type)`` pair.

    Lazy + memoized so that peers like :mod:`torch_ops` can safely
    import this helper at module top level: no work is performed until
    the tuple is actually needed.
    """
    torch_dev, torch_device_type = _detect_device()
    logger.info("torch_dev=%s, torch_device_type=%s", torch_dev, torch_device_type)
    return torch_dev, torch_device_type


@lru_cache(maxsize=1)
def current_device_spec() -> "DeviceSpec":
    """Return the :class:`DeviceSpec` for the detected device.

    Falls back to a bare ``DeviceSpec()`` (no-op / all False semantics)
    when no accelerator sub-package matches.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

    _, device_type = get_torch_device()
    spec = get_device_spec(device_type)
    if spec is None:
        if device_type != "cpu":
            logger.warning(
                "No DeviceSpec registered for %r; using fallback"
                " with no-op capabilities.",
                device_type,
            )
        return DeviceSpec()
    return spec


def normalize_device_type(device_type: str) -> str:
    """Map a torch-reported device type onto the LMCache logical device type.

    Some accelerators drive PyTorch through a compatibility shim that reuses
    another backend's torch module.  Kunlun KLX_XPU runs on ``torch_xmlir``, so its
    tensors report ``device.type == "cuda"`` even though CUDA IPC handles, CUDA
    events and the CUDA native ops are all unusable there.  Resolving such a
    tensor by its raw string would silently bind the *CUDA* spec and hand back
    a wrapper / event backend the hardware cannot honour.

    The rule is capability-driven rather than device-specific: when the running
    accelerator borrows *device_type* as its ``torch_module_name`` while
    registering a different ``device_type`` of its own, the borrowed name
    belongs to that accelerator.  Backends whose module name matches their own
    device type (CUDA, MUSA, XPU, ...) are returned unchanged, so this is a
    no-op on every conventional platform and new shim-based backends are
    picked up with zero edits here.

    Args:
        device_type: Device type as reported by torch (e.g. ``tensor.device.type``).

    Returns:
        The LMCache logical device type to use for registry lookups.
    """
    if not device_type:
        return device_type
    spec = current_device_spec()
    if spec.device_type != device_type and spec.torch_module_name == device_type:
        return spec.device_type
    return device_type
