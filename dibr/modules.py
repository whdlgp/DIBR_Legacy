import numpy as np
from typing import Literal

from .blocks import check_valid_coord, check_valid_radius
from .blocks import unproject_plane, project_plane, unproject_erp, project_erp
from .blocks import select_coord, select_valid_pixel, zbuffer_keep_nearest
from .blocks import write_image, write_depth, sample_image


def pixel2world(depth: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """src image pixels → 3D coordinates (plane).

    Args:
        depth: src depth map, shape (H, W).
        camera: src camera intrinsic matrix, shape (3, 3).

    Returns:
        3D coordinates, shape (3, H * W).
    """
    return unproject_plane(depth, camera)


def pixel2world_erp(depth: np.ndarray) -> np.ndarray:
    """src ERP image pixels → 3D coordinates.

    Args:
        depth: src ERP depth map (radius), shape (H, W).

    Returns:
        3D coordinates, shape (3, H * W).
    """
    return unproject_erp(depth)


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
    height, width = ref_image.shape[:2]

    valid = check_valid_coord(world_coord, 1e-6)
    src_idx, world_coord = select_coord(world_coord, valid)
    x, y, depth = project_plane(world_coord, camera)
    x, y, depth, src_idx = select_valid_pixel(x, y, height, width, depth, src_idx)
    x, y, depth, src_idx = zbuffer_keep_nearest(x, y, depth, src_idx, height, width)

    image = None
    if return_mode != "depth":
        image = write_image(ref_image, x, y, src_idx, height, width)

    depth_map = None
    if return_mode != "image":
        depth_map = write_depth(depth, x, y, height, width)

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
    height, width = ref_image.shape[:2]

    valid = check_valid_coord(world_coord, 0)
    idx, world_coord = select_coord(world_coord, valid)
    x, y, _ = project_plane(world_coord, camera)
    x, y, idx = select_valid_pixel(x, y, height, width, idx)

    return sample_image(ref_image, x, y, idx, height, width)


def world2pixel_forward_erp(world_coord: np.ndarray, ref_image: np.ndarray, return_mode: Literal["image", "depth", "both"] = "both") -> tuple[np.ndarray | None, np.ndarray | None]:
    """src 3D coordinates → dst ERP image pixels.

    Args:
        world_coord: src 3D coordinates, shape (3, N).
        ref_image: src ERP image for color sampling. dst has the same size.
        return_mode: Output type ("image", "depth", or "both").

    Returns:
        dst ERP image and depth map.
    """
    height, width = ref_image.shape[:2]

    valid = check_valid_radius(world_coord)
    src_idx, world_coord = select_coord(world_coord, valid)
    x, y, depth = project_erp(world_coord, height, width)
    x, y, depth, src_idx = select_valid_pixel(x, y, height, width, depth, src_idx)
    x, y, depth, src_idx = zbuffer_keep_nearest(x, y, depth, src_idx, height, width)

    image = None
    if return_mode != "depth":
        image = write_image(ref_image, x, y, src_idx, height, width)

    depth_map = None
    if return_mode != "image":
        depth_map = write_depth(depth, x, y, height, width)

    return image, depth_map


def world2pixel_inverse_erp(world_coord: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    """dst 3D coordinates → src ERP image colors.

    Args:
        world_coord: dst 3D coordinates, shape (3, N).
        ref_image: src ERP image for color sampling. dst has the same size.

    Returns:
        dst ERP image.
    """
    height, width = ref_image.shape[:2]

    valid = check_valid_radius(world_coord)
    idx, world_coord = select_coord(world_coord, valid)
    x, y, _ = project_erp(world_coord, height, width)
    x, y, idx = select_valid_pixel(x, y, height, width, idx)

    return sample_image(ref_image, x, y, idx, height, width)


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
    width, height = camera_dst_size

    valid = check_valid_coord(world_coord, 1e-6)
    src_idx, world_coord = select_coord(world_coord, valid)
    x, y, depth = project_plane(world_coord, camera)
    x, y, depth, src_idx = select_valid_pixel(x, y, height, width, depth, src_idx)
    x, y, depth, src_idx = zbuffer_keep_nearest(x, y, depth, src_idx, height, width)

    image = None
    if return_mode != "depth":
        image = write_image(ref_image, x, y, src_idx, height, width)

    depth_map = None
    if return_mode != "image":
        depth_map = write_depth(depth, x, y, height, width)

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
    width, height = camera_dst_size
    src_height, src_width = ref_image.shape[:2]

    valid = check_valid_radius(world_coord)
    idx, world_coord = select_coord(world_coord, valid)
    x, y, _ = project_erp(world_coord, src_height, src_width)
    x, y, idx = select_valid_pixel(x, y, src_height, src_width, idx)

    return sample_image(ref_image, x, y, idx, height, width)