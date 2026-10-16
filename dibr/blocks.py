import numpy as np
import cv2
from typing import Literal


def zbuffer_keep_nearest(x: np.ndarray, y: np.ndarray, depth: np.ndarray, src_idx: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the nearest-depth point for each dst pixel.

    Args:
        x: dst pixel x coordinates (int).
        y: dst pixel y coordinates (int).
        depth: depth of each point.
        src_idx: flat src pixel index of each point.
        height: dst image height.
        width: dst image width.

    Returns:
        x, y, depth, src_idx of the nearest point per pixel.
    """
    target_idx = y.astype(np.int64) * width + x

    zbuf = np.full(height * width, np.inf, dtype=depth.dtype)
    np.minimum.at(zbuf, target_idx, depth)

    winner = depth == zbuf[target_idx]

    return x[winner], y[winner], depth[winner], src_idx[winner]


def check_valid_coord(world_coord: np.ndarray, min_depth: float) -> np.ndarray:
    """Check which 3D coordinates are finite and deeper than min_depth.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        min_depth: lower bound of z (exclusive).

    Returns:
        Boolean mask, shape (N,).
    """
    return np.isfinite(world_coord).all(axis=0) & (world_coord[2] > min_depth)


def check_valid_radius(world_coord: np.ndarray) -> np.ndarray:
    """Check which 3D coordinates have a finite, positive radius.

    Args:
        world_coord: 3D coordinates, shape (3, N).

    Returns:
        Boolean mask, shape (N,).
    """
    radius = np.linalg.norm(world_coord, axis=0)

    return np.isfinite(radius) & (radius > 0)


def check_valid_pixel(x: np.ndarray, y: np.ndarray, height: int, width: int) -> np.ndarray:
    """Check which pixel coordinates are inside the image.

    Args:
        x: pixel x coordinates.
        y: pixel y coordinates.
        height: image height.
        width: image width.

    Returns:
        Boolean mask, shape (N,).
    """
    return (x >= 0) & (x < width) & (y >= 0) & (y < height)


def unproject_plane(depth: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """Plane image pixels → 3D coordinates.

    Args:
        depth: src depth map, shape (H, W).
        camera: src camera intrinsic matrix, shape (3, 3).

    Returns:
        3D coordinates, shape (3, H * W).
    """
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


def unproject_erp(depth: np.ndarray) -> np.ndarray:
    """ERP image pixels → 3D coordinates.

    Args:
        depth: src ERP depth map (radius), shape (H, W).

    Returns:
        3D coordinates, shape (3, H * W).
    """
    height, width = depth.shape

    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    lon, lat = np.meshgrid((u - 0.5) * 2 * np.pi, (0.5 - v) * np.pi)

    world_coord = np.empty((3, height * width), dtype=depth.dtype)
    world_coord[0] = (np.cos(lat) * np.sin(lon) * depth).reshape(-1)
    world_coord[1] = (-np.sin(lat) * depth).reshape(-1)
    world_coord[2] = (np.cos(lat) * np.cos(lon) * depth).reshape(-1)

    return world_coord


def project_plane(world_coord: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D coordinates → plane (pinhole) pixel coordinates.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        camera: camera intrinsic matrix, shape (3, 3).

    Returns:
        x, y (rounded pixel coordinates) and depth (z).
    """
    fx = camera[0, 0]
    fy = camera[1, 1]
    ox = camera[0, 2]
    oy = camera[1, 2]

    x = np.round(world_coord[0] * fx / world_coord[2] + ox)
    y = np.round(world_coord[1] * fy / world_coord[2] + oy)

    return x, y, world_coord[2]


def project_erp(world_coord: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D coordinates → ERP pixel coordinates.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        height: ERP image height.
        width: ERP image width.

    Returns:
        x (wrapped), y (clipped) pixel coordinates and depth (radius).
    """
    radius = np.linalg.norm(world_coord, axis=0)

    lon = np.arctan2(world_coord[0], world_coord[2])
    lat = np.arcsin(np.clip(-world_coord[1] / radius, -1, 1))

    x = np.round((lon / (2 * np.pi) + 0.5) * width - 0.5).astype(np.int32) % width
    y = np.clip(np.round((0.5 - lat / np.pi) * height - 0.5).astype(np.int32), 0, height - 1)

    return x, y, radius


def select_coord(world_coord: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select 3D coordinates by a validity mask.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        valid: boolean mask, shape (N,).

    Returns:
        Flat indices of the selected coordinates and the selected coordinates.
    """
    return np.flatnonzero(valid), world_coord[:, valid]


def select_valid_pixel(x: np.ndarray, y: np.ndarray, height: int, width: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Select points whose pixel coordinates are inside the image.

    Args:
        x: pixel x coordinates.
        y: pixel y coordinates.
        height: image height.
        width: image width.
        *arrays: additional per-point arrays to select together.

    Returns:
        x, y (int32) and the selected additional arrays.
    """
    valid = check_valid_pixel(x, y, height, width)

    return (x[valid].astype(np.int32), y[valid].astype(np.int32), *[a[valid] for a in arrays])


def write_image(ref_image: np.ndarray, x: np.ndarray, y: np.ndarray, src_idx: np.ndarray, height: int, width: int) -> np.ndarray:
    """Write src colors to dst pixels.

    Args:
        ref_image: src image, shape (H, W, C).
        x: dst pixel x coordinates.
        y: dst pixel y coordinates.
        src_idx: flat src pixel index of each point.
        height: dst image height.
        width: dst image width.

    Returns:
        dst image, shape (height, width, C).
    """
    channels = ref_image.shape[2]

    image = np.zeros((height, width, channels), dtype=ref_image.dtype)
    image[y, x] = ref_image.reshape(-1, channels)[src_idx]

    return image


def write_depth(depth: np.ndarray, x: np.ndarray, y: np.ndarray, height: int, width: int) -> np.ndarray:
    """Write depths to dst pixels.

    Args:
        depth: depth of each point.
        x: dst pixel x coordinates.
        y: dst pixel y coordinates.
        height: dst depth map height.
        width: dst depth map width.

    Returns:
        dst depth map, shape (height, width).
    """
    depth_map = np.zeros((height, width), dtype=np.float32)
    depth_map[y, x] = depth

    return depth_map


def sample_image(ref_image: np.ndarray, x: np.ndarray, y: np.ndarray, idx: np.ndarray, height: int, width: int) -> np.ndarray:
    """Sample src colors at (x, y) and write them to dst pixels.

    Args:
        ref_image: src image, shape (H, W, C).
        x: src pixel x coordinates.
        y: src pixel y coordinates.
        idx: flat dst pixel index of each point.
        height: dst image height.
        width: dst image width.

    Returns:
        dst image, shape (height, width, C).
    """
    channels = ref_image.shape[2]

    image = np.zeros((height, width, channels), dtype=ref_image.dtype)
    image.reshape(-1, channels)[idx] = ref_image[y, x]

    return image


def make_rt(theta_x: float, theta_y: float, theta_z: float, tx: float, ty: float, tz: float) -> tuple[np.ndarray, np.ndarray]:
    """Make rotation matrix and translation vector.

    Args:
        theta_x: rotation about the x axis (radians).
        theta_y: rotation about the y axis (radians).
        theta_z: rotation about the z axis (radians).
        tx: translation along x.
        ty: translation along y.
        tz: translation along z.

    Returns:
        R (3, 3) and t (3, 1).
    """
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


def apply_rt(world_coord: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply R|t to 3D coordinates.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).

    Returns:
        Transformed 3D coordinates, shape (3, N).
    """
    return R @ (world_coord - t)


def apply_rt_inverse(world_coord: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply inverse R|t to 3D coordinates.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).

    Returns:
        Transformed 3D coordinates, shape (3, N).
    """
    return R.T @ world_coord + t


def filter_depth(depth: np.ndarray, filter_type: Literal["morph", "median"]) -> np.ndarray:
    """Fill cracks in a warped depth map.

    Args:
        depth: depth map, shape (H, W).
        filter_type: Filter type ("morph" or "median").

    Returns:
        Filtered depth map, shape (H, W).
    """
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