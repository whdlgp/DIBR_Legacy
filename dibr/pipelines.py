import numpy as np
import cv2
from typing import Literal
from util.show import depth_to_show, show

from .device import get_blocks
from .modules import pixel2world, pixel2world_erp
from .modules import world2pixel_forward, world2pixel_inverse
from .modules import world2pixel_forward_erp, world2pixel_inverse_erp
from .modules import world2pixel_forward_erp2plane, world2pixel_inverse_erp2plane


def forward_mapping(depth: np.ndarray, image: np.ndarray, camera_src: np.ndarray, camera_dst: np.ndarray, R: np.ndarray, t: np.ndarray,
                    return_mode: Literal["image", "depth", "both"] = "both", verbose: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """src plane image → dst plane image by forward mapping.

    Args:
        depth: src depth map, shape (H, W).
        image: src image, shape (H, W, C).
        camera_src: src camera intrinsic matrix, shape (3, 3).
        camera_dst: dst camera intrinsic matrix, shape (3, 3).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).
        return_mode: Output type ("image", "depth", or "both").
        verbose: Print progress.

    Returns:
        dst image and depth map.
    """
    bl = get_blocks()

    verbose and print("Source image → 3D World coordinates")
    world_coord = pixel2world(depth, camera_src)

    verbose and print("Apply R|t")
    world_coord = bl.apply_rt(world_coord, R, t)

    verbose and print("Forward mapping")
    image_warped, depth_warped = world2pixel_forward(world_coord, image, camera_dst, return_mode=return_mode)

    verbose and print("Done.")

    return image_warped, depth_warped


def inverse_mapping(depth: np.ndarray, image: np.ndarray, camera_src: np.ndarray, camera_dst: np.ndarray, R: np.ndarray, t: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """src plane image → dst plane image by inverse mapping.

    Args:
        depth: src depth map, shape (H, W).
        image: src image, shape (H, W, C).
        camera_src: src camera intrinsic matrix, shape (3, 3).
        camera_dst: dst camera intrinsic matrix, shape (3, 3).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).
        verbose: Print progress.

    Returns:
        dst image and filtered dst depth map.
    """
    bl = get_blocks()
    
    # Do forward mapping
    _, depth_warped = forward_mapping(depth, image, camera_src, camera_dst, R, t, 'depth', verbose)

    verbose and print("Median filtering")
    depth_filt = bl.filter_depth(depth_warped, 'median')

    verbose and print("Reproject depth to 3D World coordinates")
    world_coord = pixel2world(depth_filt, camera_dst)

    verbose and print("Apply reverse R|t")
    world_coord = bl.apply_rt_inverse(world_coord, R, t)

    verbose and print("Inverse mapping")
    image_inverse = world2pixel_inverse(world_coord, image, camera_src)

    verbose and print("Done.")

    return image_inverse, depth_filt


def forward_mapping_erp(depth: np.ndarray, image: np.ndarray, R: np.ndarray, t: np.ndarray,
                        return_mode: Literal["image", "depth", "both"] = "both", verbose: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """src ERP image → dst ERP image by forward mapping.

    Args:
        depth: src ERP depth map (radius), shape (H, W).
        image: src ERP image, shape (H, W, C).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).
        return_mode: Output type ("image", "depth", or "both").
        verbose: Print progress.

    Returns:
        dst ERP image and depth map.
    """
    bl = get_blocks()
    
    verbose and print("Source image → 3D World coordinates")
    world_coord = pixel2world_erp(depth)

    verbose and print("Apply R|t")
    world_coord = bl.apply_rt(world_coord, R, t)

    verbose and print("Forward mapping")
    image_warped, depth_warped = world2pixel_forward_erp(world_coord, image, return_mode=return_mode)

    verbose and print("Done.")

    return image_warped, depth_warped


def inverse_mapping_erp(depth: np.ndarray, image: np.ndarray, R: np.ndarray, t: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """src ERP image → dst ERP image by inverse mapping.

    Args:
        depth: src ERP depth map (radius), shape (H, W).
        image: src ERP image, shape (H, W, C).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).
        verbose: Print progress.

    Returns:
        dst ERP image and filtered dst depth map.
    """
    bl = get_blocks()
    
    _, depth_warped = forward_mapping_erp(depth, image, R, t, "depth", verbose)

    verbose and print("Morph filtering")
    depth_filt = bl.filter_depth(depth_warped, "morph")

    verbose and print("Reproject depth to 3D World coordinates")
    world_coord = pixel2world_erp(depth_filt)

    verbose and print("Apply reverse R|t")
    world_coord = bl.apply_rt_inverse(world_coord, R, t)

    verbose and print("Inverse mapping")
    image_inverse = world2pixel_inverse_erp(world_coord, image)

    verbose and print("Done.")

    return image_inverse, depth_filt


def forward_mapping_erp2plane(depth: np.ndarray, image: np.ndarray, camera_dst: np.ndarray, camera_dst_size: tuple[int, int], R: np.ndarray, t: np.ndarray,
                              return_mode: Literal["image", "depth", "both"] = "both", verbose: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """src ERP image → dst plane image by forward mapping.

    Args:
        depth: src ERP depth map (radius), shape (H, W).
        image: src ERP image, shape (H, W, C).
        camera_dst: dst camera intrinsic matrix, shape (3, 3).
        camera_dst_size: dst image size as (width, height).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).
        return_mode: Output type ("image", "depth", or "both").
        verbose: Print progress.

    Returns:
        dst plane image and depth map.
    """
    bl = get_blocks()
    
    verbose and print("Source image → 3D World coordinates")
    world_coord = pixel2world_erp(depth)

    verbose and print("Apply R|t")
    world_coord = bl.apply_rt(world_coord, R, t)

    verbose and print("Forward mapping")
    image_warped, depth_warped = world2pixel_forward_erp2plane(world_coord, image, camera_dst, camera_dst_size, return_mode=return_mode)

    verbose and print("Done.")

    return image_warped, depth_warped


def inverse_mapping_erp2plane(depth: np.ndarray, image: np.ndarray, camera_dst: np.ndarray, camera_dst_size: tuple[int, int], R: np.ndarray, t: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """src ERP image → dst plane image by inverse mapping.

    Args:
        depth: src ERP depth map (radius), shape (H, W).
        image: src ERP image, shape (H, W, C).
        camera_dst: dst camera intrinsic matrix, shape (3, 3).
        camera_dst_size: dst image size as (width, height).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).
        verbose: Print progress.

    Returns:
        dst plane image and filtered dst depth map.
    """
    bl = get_blocks()
    
    verbose and print("Forward mapping")
    _, depth_warped = forward_mapping_erp2plane(depth, image, camera_dst, camera_dst_size, R, t, "depth")

    verbose and print("Filter depth")
    depth_filt = bl.filter_depth(depth_warped, "morph")

    verbose and print("Reproject depth to 3D World coordinates")
    world_coord = pixel2world(depth_filt, camera_dst)

    verbose and print("Apply reverse R|t")
    world_coord = bl.apply_rt_inverse(world_coord, R, t)

    verbose and print("Inverse mapping")
    image_inverse = world2pixel_inverse_erp2plane(world_coord, image, camera_dst_size)

    verbose and print("Done.")

    return image_inverse, depth_filt


def alpha_blending(image_ref1: np.ndarray, image_ref2: np.ndarray, depth_dst1: np.ndarray, depth_dst2: np.ndarray, alpha: float) -> np.ndarray:
    """Blend two warped images, filling occluded areas from the other image.

    Args:
        image_ref1: first warped image, shape (H, W, C).
        image_ref2: second warped image, shape (H, W, C).
        depth_dst1: depth map of the first warped image, shape (H, W).
        depth_dst2: depth map of the second warped image, shape (H, W).
        alpha: blending weight of the second image (0 to 1).

    Returns:
        Blended image (uint8), shape (H, W, C).
    """
    # Find occluded area
    occ1 = depth_dst1 <= 10
    occ2 = depth_dst2 <= 10

    # Expend(matting) occluded area
    kernel1 = np.array([[1, 1, 0]], dtype=np.uint8)
    kernel2 = np.array([[0, 1, 1]], dtype=np.uint8)
    occ1 = cv2.dilate(occ1.astype(np.uint8), kernel1).astype(bool)
    occ2 = cv2.dilate(occ2.astype(np.uint8), kernel2).astype(bool)

    # alpha blending
    blended_image = np.zeros_like(image_ref1)

    valid_both = ~occ1 & ~occ2
    valid1 = ~occ1 & occ2
    valid2 = occ1 & ~occ2

    blended_image[valid_both] = (1 - alpha) * image_ref1[valid_both] + alpha * image_ref2[valid_both]
    blended_image[valid1] = image_ref1[valid1]
    blended_image[valid2] = image_ref2[valid2]

    return blended_image.astype(np.uint8)


def show_mapping_comparison(images: list[np.ndarray], depths: list[np.ndarray], show_width: int = 1280, title: str = "Mapping", axis: Literal["vertical", "horizontal"] = "vertical") -> None:
    """Show warped images and their depth maps side by side.

    Args:
        images: images to compare.
        depths: depth maps of the images.
        show_width: width of the shown window.
        title: window title.
        axis: Concatenation axis of the images ("vertical" or "horizontal").
    """
    concat = cv2.vconcat if axis == "vertical" else cv2.hconcat
    concat_axis = cv2.hconcat if axis == "vertical" else cv2.vconcat

    images = concat(images)
    depths = concat(depths)

    depth_show = depth_to_show(depths)

    show_img = concat_axis([images, depth_show])
    show(show_img, show_width, title)