import numpy as np
import cv2
from typing import Literal
from util.show import depth_to_show, show


def pixel2world(depth, camera):
    height, width = depth.shape

    fx = camera[0, 0]
    fy = camera[1, 1]
    ox = camera[0, 2]
    oy = camera[1, 2]

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    world_coord = np.empty((3, height * width), dtype=depth.dtype)
    world_coord[0] = ((x - ox) * depth / fx).reshape(-1)
    world_coord[1] = ((y - oy) * depth / fy).reshape(-1)
    world_coord[2] = depth.reshape(-1)

    return world_coord


def world2pixel_forward(world_coord, ref_image, camera, return_mode: Literal["image", "depth", "both"] = "both") -> tuple[np.ndarray | None, np.ndarray | None]:
    """src 3D coordinates → dst image pixels.

    Args:
        world_coord: src 3D coordinates.
        ref_image: src image for color sampling.
        camera: dst camera parameters.
        return_mode: Output type ("image", "depth", or "both").

    Returns:
        dst image and depth map.
    """
    height, width = ref_image.shape[:2]

    fx = camera[0, 0]
    fy = camera[1, 1]
    ox = camera[0, 2]
    oy = camera[1, 2]

    valid = np.isfinite(world_coord[2]) & (world_coord[2] > 0)

    src_idx = np.flatnonzero(valid)
    world_coord = world_coord[:, valid]

    x = np.round(world_coord[0] * fx / world_coord[2] + ox).astype(np.int32)
    y = np.round(world_coord[1] * fy / world_coord[2] + oy).astype(np.int32)
    depth = world_coord[2]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    x = x[valid]
    y = y[valid]
    depth = depth[valid]
    src_idx = src_idx[valid]

    order = np.argsort(depth)

    x = x[order]
    y = y[order]
    depth = depth[order]
    src_idx = src_idx[order]

    target_idx = y * width + x
    _, first = np.unique(target_idx, return_index=True)

    x = x[first]
    y = y[first]
    depth = depth[first]
    src_idx = src_idx[first]

    im_another_point = None
    depth_another_point = None

    if return_mode != "depth":
        im_another_point = np.zeros_like(ref_image)
        im_another_point[y, x] = ref_image.reshape(-1, ref_image.shape[2])[src_idx]

    if return_mode != "image":
        depth_another_point = np.zeros((height, width), dtype=np.float32)
        depth_another_point[y, x] = depth

    return im_another_point, depth_another_point


def world2pixel_inverse(world_coord, ref_image, camera):
    """dst 3D coordinates → src image colors.

    Args:
        world_coord: dst 3D coordinates.
        ref_image: src image for color sampling.
        camera: src camera parameters.

    Returns:
        dst image.
    """
    height, width = ref_image.shape[:2]

    fx, fy = camera[0, 0], camera[1, 1]
    ox, oy = camera[0, 2], camera[1, 2]

    valid_coord = np.isfinite(world_coord).all(axis=0) & (world_coord[2] > 0)
    idx = np.flatnonzero(valid_coord)
    world_coord = world_coord[:, valid_coord]

    x = np.round(world_coord[0] * fx / world_coord[2] + ox).astype(np.int32)
    y = np.round(world_coord[1] * fy / world_coord[2] + oy).astype(np.int32)

    valid_pixel = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    idx = idx[valid_pixel]

    im_inverse = np.zeros_like(ref_image)
    im_inverse.reshape(-1, ref_image.shape[2])[idx] = ref_image[y[valid_pixel], x[valid_pixel]]

    return im_inverse


def pixel2world_erp(depth):
    height, width = depth.shape

    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    lon, lat = np.meshgrid((u - 0.5) * 2 * np.pi, (0.5 - v) * np.pi)

    world_coord = np.empty((3, height * width), dtype=depth.dtype)
    world_coord[0] = (np.cos(lat) * np.sin(lon) * depth).reshape(-1)
    world_coord[1] = (-np.sin(lat) * depth).reshape(-1)
    world_coord[2] = (np.cos(lat) * np.cos(lon) * depth).reshape(-1)

    return world_coord


def world2pixel_forward_erp(world_coord, ref_image, return_mode="both"):
    height, width = ref_image.shape[:2]

    radius = np.linalg.norm(world_coord, axis=0)
    valid = np.isfinite(radius) & (radius > 0)

    src_idx = np.flatnonzero(valid)
    world_coord = world_coord[:, valid]
    depth = radius[valid]

    lon = np.arctan2(world_coord[0], world_coord[2])
    lat = np.arcsin(np.clip(-world_coord[1] / depth, -1, 1))

    x = np.round((lon / (2 * np.pi) + 0.5) * width - 0.5).astype(np.int32) % width
    y = np.clip(np.round((0.5 - lat / np.pi) * height - 0.5).astype(np.int32), 0, height - 1)

    order = np.argsort(depth)

    x = x[order]
    y = y[order]
    depth = depth[order]
    src_idx = src_idx[order]

    target_idx = y * width + x
    _, first = np.unique(target_idx, return_index=True)

    x = x[first]
    y = y[first]
    depth = depth[first]
    src_idx = src_idx[first]

    im_another_point = None
    depth_another_point = None

    if return_mode != "depth":
        im_another_point = np.zeros_like(ref_image)
        im_another_point[y, x] = ref_image.reshape(-1, ref_image.shape[2])[src_idx]

    if return_mode != "image":
        depth_another_point = np.zeros((height, width), dtype=np.float32)
        depth_another_point[y, x] = depth

    return im_another_point, depth_another_point


def world2pixel_inverse_erp(world_coord, ref_image):
    height, width = ref_image.shape[:2]

    radius = np.linalg.norm(world_coord, axis=0)
    valid = np.isfinite(radius) & (radius > 0)

    idx = np.flatnonzero(valid)
    world_coord = world_coord[:, valid]
    radius = radius[valid]

    lon = np.arctan2(world_coord[0], world_coord[2])
    lat = np.arcsin(np.clip(-world_coord[1] / radius, -1, 1))

    x = np.round((lon / (2 * np.pi) + 0.5) * width - 0.5).astype(np.int32) % width
    y = np.clip(np.round((0.5 - lat / np.pi) * height - 0.5).astype(np.int32), 0, height - 1)

    im_inverse = np.zeros_like(ref_image)
    im_inverse.reshape(-1, ref_image.shape[2])[idx] = ref_image[y, x]

    return im_inverse


def make_rt(theta_x, theta_y, theta_z, tx, ty, tz):
    rot_x = np.array([
        [1, 0              ,  0              ],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x),  np.cos(theta_x)]
    ])

    rot_y = np.array([
        [ np.cos(theta_y), 0, np.sin(theta_y)],
        [0               , 1, 0              ],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0              ,  0              , 1]
    ])

    R = rot_z @ rot_y @ rot_x
    t = np.array([[tx], [ty], [tz]], dtype=np.float64)

    return R, t


def apply_rt(world_coord, R, t):
    return R @ (world_coord - t)


def apply_rt_inverse(world_coord, R, t):
    return R.T @ world_coord + t


def filter_depth(depth, filter_type: Literal["morph", "median"]):
    if filter_type == "morph":
        se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        depth = cv2.dilate(depth, se1)
        depth = cv2.erode(depth, se2)
        depth = cv2.erode(depth, se3)
    elif filter_type == "median":
        depth = cv2.medianBlur(depth, 5)

    return depth


def forward_mapping(depth, image, camera_src, camera_dst, R, t,
                    return_mode: Literal["image", "depth", "both"] = "both", verbose=False):
    verbose and print("Source image → 3D World coordinates")
    world_coord = pixel2world(depth, camera_src)

    verbose and print("Apply R|t")
    world_coord = apply_rt(world_coord, R, t)

    verbose and print("Forward mapping")
    image_warped, depth_warped = world2pixel_forward(world_coord, image, camera_dst, return_mode=return_mode)

    verbose and print("Done.")

    return image_warped, depth_warped


def inverse_mapping(depth, image, camera_src, camera_dst, R, t, verbose=False):
    # Do forward mapping
    _, depth_warped = forward_mapping(depth, image, camera_src, camera_dst, R, t, 'depth', verbose)

    verbose and print("Median filtering")
    depth_filt = filter_depth(depth_warped, 'median')

    verbose and print("Reproject depth to 3D World coordinates")
    world_coord = pixel2world(depth_filt, camera_dst)

    verbose and print("Apply reverse R|t")
    world_coord = apply_rt_inverse(world_coord, R, t)

    verbose and print("Inverse mapping")
    image_inverse = world2pixel_inverse(world_coord, image, camera_src)

    verbose and print("Done.")

    return image_inverse, depth_filt


def forward_mapping_erp(depth, image, R, t, return_mode="both", verbose=False):
    verbose and print("Source image → 3D World coordinates")
    world_coord = pixel2world_erp(depth)
    world_coord[:, depth.reshape(-1) <= 0] = np.nan

    verbose and print("Apply R|t")
    world_coord = apply_rt(world_coord, R, t)

    verbose and print("Forward mapping")
    image_warped, depth_warped = world2pixel_forward_erp(world_coord, image, return_mode=return_mode)

    verbose and print("Done.")

    return image_warped, depth_warped


def inverse_mapping_erp(depth, image, R, t, verbose=False):
    _, depth_warped = forward_mapping_erp(depth, image, R, t, "depth", verbose)

    verbose and print("Morph filtering")
    depth_filt = filter_depth(depth_warped, "morph")

    verbose and print("Reproject depth to 3D World coordinates")
    world_coord = pixel2world_erp(depth_filt)
    world_coord[:, depth_filt.reshape(-1) <= 0] = np.nan

    verbose and print("Apply reverse R|t")
    world_coord = apply_rt_inverse(world_coord, R, t)

    verbose and print("Inverse mapping")
    image_inverse = world2pixel_inverse_erp(world_coord, image)

    verbose and print("Done.")

    return image_inverse, depth_filt


def alpha_blending(image_ref1, image_ref2, depth_dst1, depth_dst2, alpha):
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


def show_mapping_comparison(images, depths, show_width=1280, title="Mapping", axis: Literal["vertical", "horizontal"] = "vertical"):
    concat = cv2.vconcat if axis == "vertical" else cv2.hconcat
    concat_axis = cv2.hconcat if axis == "vertical" else cv2.vconcat

    images = concat(images)
    depths = concat(depths)

    depth_show = depth_to_show(depths)

    show_img = concat_axis([images, depth_show])
    show(show_img, show_width, title)
