from __future__ import annotations

"""Minimal single-ndarray visualization helpers.

Functions
---------
- squeeze_to_image(arr): reduce common bioimaging array shapes to 2D
- normalize_to_uint8(img): scale to [0, 255] for display
- show_image(arr, ...): squeeze + normalize and display/save with matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np


def squeeze_to_image(arr: np.ndarray) -> np.ndarray:
    """Return a 2D view from an ndarray for display.

    Heuristics:
    - (H, W): return as-is
    - (C, H, W): first channel
    - (N, C, H, W): first batch, first channel
    - (N, H, W, C): first batch, mean over channels (or first if single)
    - (H, W, C): mean over channels (or single if 1)
    - (Z, H, W): max projection over Z
    For higher dims, iteratively max-project until 2D.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a

    if a.ndim == 3:
        # (C, H, W) or (Z, H, W) or (H, W, C)
        if a.shape[0] <= 4 and a.shape[0] != a.shape[-1]:
            return a[0]  # (C, H, W)
        if a.shape[-1] <= 4 and a.shape[-1] != a.shape[0]:
            return a[..., 0] if a.shape[-1] == 1 else a.mean(axis=-1)  # (H, W, C)
        return a.max(axis=0)  # (Z, H, W)

    if a.ndim == 4:
        # (N, C, H, W) or (N, H, W, C)
        if a.shape[1] <= 8 and a.shape[1] != a.shape[-1]:
            return a[0, 0]  # (N, C, H, W)
        if a.shape[-1] <= 8:
            first = a[0]  # (N, H, W, C)
            return first[..., 0] if first.shape[-1] == 1 else first.mean(axis=-1)
        return squeeze_to_image(a.max(axis=0))  # fallback
    # Generic fallback
    while a.ndim > 2:
        a = a.max(axis=0)
    return a


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize a numeric array to uint8 [0, 255] for display."""
    x = np.asarray(img)
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)
    vmin = x[finite].min()
    vmax = x[finite].max()
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    scaled = (x - vmin) / (vmax - vmin)
    scaled[~finite] = 0
    return np.clip((scaled * 255.0).round(), 0, 255).astype(np.uint8)


def show_image(
    arr: np.ndarray,
    *,
    cmap: str = "gray",
    save_path: str | None = None,
    show: bool = True,
):
    """Display a single ndarray as an image using matplotlib.

    Squeezes to 2D and displays with suitable scaling. Returns the Figure.
    """
    img2d = squeeze_to_image(arr)
    fig, ax = plt.subplots(figsize=(6, 6))
    if np.issubdtype(img2d.dtype, np.floating):
        ax.imshow(
            img2d,
            cmap=cmap,
            vmin=float(np.nanmin(img2d)),
            vmax=float(np.nanmax(img2d)),
        )
    else:
        ax.imshow(normalize_to_uint8(img2d), cmap=cmap)
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    return fig
