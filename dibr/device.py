import warnings
from types import ModuleType

state = {"device": "numpy", "arch": None}


def set_device(device: str = "auto") -> str:
    """Select the backend of blocks. Call it once at the start.

    Args:
        device: "numpy", "auto", "cpu", "cuda", "vulkan" or "metal".

    Returns:
        Selected device ("numpy" or the Taichi arch in use).
    """
    if device == "numpy":
        state["device"] = "numpy"
        return "numpy"

    if device not in ["auto", "cpu", "cuda", "vulkan", "metal"]:
        raise ValueError(f"Unknown device '{device}'.")

    # Taichi can be initialized only once per process
    if state["arch"] is None:
        try:
            from . import blocks_taichi
            state["arch"] = blocks_taichi.init("gpu" if device == "auto" else device)
        except (ImportError, RuntimeError) as e:
            if device != "auto":
                raise
            warnings.warn(f"No Taichi GPU backend ({e}), using numpy.")
            state["device"] = "numpy"
            return "numpy"
    elif device == "auto" and state["arch"] == "cpu":
        state["device"] = "numpy"
        return "numpy"
    elif device != "auto" and device != state["arch"]:
        raise RuntimeError(f"Taichi is already initialized with '{state['arch']}'.")

    state["device"] = state["arch"]

    return state["arch"]


def get_device() -> str:
    """Get the selected device ("numpy" or the Taichi arch in use)."""
    return state["device"]


def get_blocks() -> ModuleType:
    """Get the blocks module of the selected device."""
    if state["device"] == "numpy":
        from . import blocks
        return blocks

    from . import blocks_taichi
    return blocks_taichi
