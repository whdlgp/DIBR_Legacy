import numpy as np
from typing import Literal

from .device import get_blocks


def pixel2world(depth: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """src image pixels → 3D coordinates (plane).

    Args:
        depth: src depth map, shape (H, W).
        camera: src camera intrinsic matrix, shape (3, 3).

    Returns:
        3D coordinates, shape (3, H * W).
    """
    return get_blocks().unproject_plane(depth, camera)


def pixel2world_erp(depth: np.ndarray) -> np.ndarray:
    """src ERP image pixels → 3D coordinates.

    Args:
        depth: src ERP depth map (radius), shape (H, W).

    Returns:
        3D coordinates, shape (3, H * W).
    """
    return get_blocks().unproject_erp(depth)


def world2pixel_forward(world_coord: np.ndarray, ref_image: np.ndarray, camera: np.ndarray, return_mode: Literal["image", "depth", "both"] = "both") -> tuple[np.ndarray | None, np.ndarray | None]:
    """src 3D coordinates → dst image pixels.

    Args:
        world_coord: src 3D coordinates, shape (3, N).
        ref_image: src image for color sampling.
        camera: dst camera parameters.
        return_mode: Output type ("image", "depth", or "both").

    Returns:
        dst image and depth map.
    """
    bl = get_blocks()
    height, width = ref_image.shape[:2]

    src_idx, x, y, depth = bl.project_plane(world_coord, camera, height, width, 1e-6)
    x, y, depth, src_idx = bl.zbuffer_keep_nearest(x, y, depth, src_idx, height, width)

    image = None
    if return_mode != "depth":
        image = bl.render_image_forward(ref_image, x, y, src_idx, height, width)

    depth_map = None
    if return_mode != "image":
        depth_map = bl.render_depth_forward(depth, x, y, height, width)

    return image, depth_map


def world2pixel_inverse(world_coord: np.ndarray, ref_image: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """dst 3D coordinates → src image colors.

    Args:
        world_coord: dst 3D coordinates, shape (3, N).
        ref_image: src image for color sampling.
        camera: src camera parameters.

    Returns:
        dst image.
    """
    bl = get_blocks()
    height, width = ref_image.shape[:2]

    idx, x, y, _ = bl.project_plane(world_coord, camera, height, width, 0)

    return bl.render_image_inverse(ref_image, x, y, idx, height, width)


def world2pixel_forward_erp(world_coord: np.ndarray, ref_image: np.ndarray, return_mode: Literal["image", "depth", "both"] = "both") -> tuple[np.ndarray | None, np.ndarray | None]:
    """src 3D coordinates → dst ERP image pixels.

    Args:
        world_coord: src 3D coordinates, shape (3, N).
        ref_image: src ERP image for color sampling. dst has the same size.
        return_mode: Output type ("image", "depth", or "both").

    Returns:
        dst ERP image and depth map.
    """
    bl = get_blocks()
    height, width = ref_image.shape[:2]

    src_idx, x, y, depth = bl.project_erp(world_coord, height, width)
    x, y, depth, src_idx = bl.zbuffer_keep_nearest(x, y, depth, src_idx, height, width)

    image = None
    if return_mode != "depth":
        image = bl.render_image_forward(ref_image, x, y, src_idx, height, width)

    depth_map = None
    if return_mode != "image":
        depth_map = bl.render_depth_forward(depth, x, y, height, width)

    return image, depth_map


def world2pixel_inverse_erp(world_coord: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    """dst 3D coordinates → src ERP image colors.

    Args:
        world_coord: dst 3D coordinates, shape (3, N).
        ref_image: src ERP image for color sampling. dst has the same size.

    Returns:
        dst ERP image.
    """
    bl = get_blocks()
    height, width = ref_image.shape[:2]

    idx, x, y, _ = bl.project_erp(world_coord, height, width)

    return bl.render_image_inverse(ref_image, x, y, idx, height, width)


def world2pixel_forward_erp2plane(world_coord: np.ndarray, ref_image: np.ndarray, camera: np.ndarray, camera_dst_size: tuple[int, int], return_mode: Literal["image", "depth", "both"] = "both") -> tuple[np.ndarray | None, np.ndarray | None]:
    """src 3D coordinates (from ERP) → dst plane image pixels.

    Args:
        world_coord: src 3D coordinates, shape (3, N).
        ref_image: src ERP image for color sampling.
        camera: dst camera parameters.
        camera_dst_size: dst image size as (width, height).
        return_mode: Output type ("image", "depth", or "both").

    Returns:
        dst plane image and depth map.
    """
    bl = get_blocks()
    width, height = camera_dst_size

    src_idx, x, y, depth = bl.project_plane(world_coord, camera, height, width, 1e-6)
    x, y, depth, src_idx = bl.zbuffer_keep_nearest(x, y, depth, src_idx, height, width)

    image = None
    if return_mode != "depth":
        image = bl.render_image_forward(ref_image, x, y, src_idx, height, width)

    depth_map = None
    if return_mode != "image":
        depth_map = bl.render_depth_forward(depth, x, y, height, width)

    return image, depth_map


def world2pixel_inverse_erp2plane(world_coord: np.ndarray, ref_image: np.ndarray, camera_dst_size: tuple[int, int]) -> np.ndarray:
    """dst 3D coordinates → src ERP image colors.

    Args:
        world_coord: dst 3D coordinates, shape (3, N).
        ref_image: src ERP image for color sampling.
        camera_dst_size: dst image size as (width, height).

    Returns:
        dst plane image.
    """
    bl = get_blocks()
    width, height = camera_dst_size
    src_height, src_width = ref_image.shape[:2]

    idx, x, y, _ = bl.project_erp(world_coord, src_height, src_width)

    return bl.render_image_inverse(ref_image, x, y, idx, height, width)
