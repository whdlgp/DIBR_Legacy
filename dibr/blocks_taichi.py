import sys
import numpy as np
import cv2
import taichi as ti
from typing import Literal


def init(arch: str) -> str:
    """Initialize Taichi.

    Args:
        arch: "cpu", "cuda", "vulkan", "metal" or "gpu" (first available GPU).

    Returns:
        Name of the arch in use.
    """
    archs = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "metal": ti.metal}

    # ti.gpu also tries OpenGL, which crashes on machines without a GPU
    if arch == "gpu":
        candidates = ["metal"] if sys.platform == "darwin" else ["cuda", "vulkan"]
    else:
        candidates = [arch]

    for name in candidates:
        ti.init(arch=archs[name], fast_math=False, offline_cache=True, unrolling_limit=0, log_level=ti.ERROR)
        if ti.lang.impl.current_cfg().arch == archs[name]:
            return name
        ti.reset()

    raise RuntimeError(f"Taichi arch '{arch}' is not available (tried: {', '.join(candidates)}).")


def as_array(x: np.ndarray | ti.Ndarray, dtype: type) -> np.ndarray | ti.Ndarray:
    """Make numpy input contiguous in dtype. Taichi arrays are returned as they are."""
    if isinstance(x, ti.Ndarray):
        return x
    return np.ascontiguousarray(x, dtype=dtype)


def ti_dtype(image: np.ndarray):
    """Taichi dtype of an image."""
    return {np.uint8: ti.u8, np.uint16: ti.u16, np.float32: ti.f32}[image.dtype.type]


@ti.kernel
def zbuffer_depth_kernel(x: ti.types.ndarray(), y: ti.types.ndarray(), depth: ti.types.ndarray(), src_idx: ti.types.ndarray(), zbuf: ti.types.ndarray(), width: ti.i32):
    # depth > 0, so float32 bits as int32 keep the order of depth
    for i in range(x.shape[0]):
        if src_idx[i] >= 0:
            ti.atomic_min(zbuf[y[i] * width + x[i]], ti.bit_cast(depth[i], ti.i32))


@ti.kernel
def zbuffer_winner_kernel(x: ti.types.ndarray(), y: ti.types.ndarray(), depth: ti.types.ndarray(), src_idx: ti.types.ndarray(), zbuf: ti.types.ndarray(), winner: ti.types.ndarray(), width: ti.i32):
    # When depths tie the last point wins, same as numpy assignment
    for i in range(x.shape[0]):
        if src_idx[i] >= 0:
            target_idx = y[i] * width + x[i]
            if ti.bit_cast(depth[i], ti.i32) == zbuf[target_idx]:
                ti.atomic_max(winner[target_idx], i)


@ti.kernel
def zbuffer_select_kernel(x: ti.types.ndarray(), y: ti.types.ndarray(), depth: ti.types.ndarray(), src_idx: ti.types.ndarray(), winner: ti.types.ndarray(), x_out: ti.types.ndarray(), y_out: ti.types.ndarray(), depth_out: ti.types.ndarray(), src_idx_out: ti.types.ndarray(), width: ti.i32):
    for i in range(x.shape[0]):
        x_out[i] = -1
        y_out[i] = -1
        depth_out[i] = 0.0
        src_idx_out[i] = -1
        if src_idx[i] >= 0:
            if winner[y[i] * width + x[i]] == i:
                x_out[i] = x[i]
                y_out[i] = y[i]
                depth_out[i] = depth[i]
                src_idx_out[i] = src_idx[i]


def zbuffer_keep_nearest(x: ti.Ndarray, y: ti.Ndarray, depth: ti.Ndarray, src_idx: ti.Ndarray, height: int, width: int) -> tuple[ti.Ndarray, ti.Ndarray, ti.Ndarray, ti.Ndarray]:
    """Keep only the nearest-depth point for each dst pixel.

    Args:
        x: dst pixel x coordinates (int), shape (N,).
        y: dst pixel y coordinates (int), shape (N,).
        depth: depth of each point, shape (N,).
        src_idx: flat src pixel index of each point, shape (N,). -1 for invalid points.
        height: dst image height.
        width: dst image width.

    Returns:
        x, y, depth, src_idx of the nearest point per pixel, shape (N,). Other points are -1 (depth 0).
    """
    x = as_array(x, np.int32)
    y = as_array(y, np.int32)
    depth = as_array(depth, np.float32)
    src_idx = as_array(src_idx, np.int32)

    zbuf = ti.ndarray(ti.i32, shape=(height * width,))
    winner = ti.ndarray(ti.i32, shape=(height * width,))
    zbuf.fill(int(np.float32(np.inf).view(np.int32)))
    winner.fill(-1)

    x_out = ti.ndarray(ti.i32, shape=x.shape)
    y_out = ti.ndarray(ti.i32, shape=x.shape)
    depth_out = ti.ndarray(ti.f32, shape=x.shape)
    src_idx_out = ti.ndarray(ti.i32, shape=x.shape)

    zbuffer_depth_kernel(x, y, depth, src_idx, zbuf, width)
    zbuffer_winner_kernel(x, y, depth, src_idx, zbuf, winner, width)
    zbuffer_select_kernel(x, y, depth, src_idx, winner, x_out, y_out, depth_out, src_idx_out, width)

    return x_out, y_out, depth_out, src_idx_out


@ti.kernel
def unproject_plane_kernel(depth: ti.types.ndarray(), world_coord: ti.types.ndarray(), fx: ti.f32, fy: ti.f32, ox: ti.f32, oy: ti.f32):
    height = depth.shape[0]
    width = depth.shape[1]

    for y, x in ti.ndrange(height, width):
        i = y * width + x
        world_coord[0, i] = (x - ox) * depth[y, x] / fx
        world_coord[1, i] = (y - oy) * depth[y, x] / fy
        world_coord[2, i] = depth[y, x]


def unproject_plane(depth: np.ndarray, camera: np.ndarray) -> ti.Ndarray:
    """Plane image pixels → 3D coordinates.

    Args:
        depth: src depth map, shape (H, W).
        camera: src camera intrinsic matrix, shape (3, 3).

    Returns:
        3D coordinates, shape (3, H * W).
    """
    height, width = depth.shape

    world_coord = ti.ndarray(ti.f32, shape=(3, height * width))
    unproject_plane_kernel(as_array(depth, np.float32), world_coord, camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2])

    return world_coord


@ti.kernel
def unproject_erp_kernel(depth: ti.types.ndarray(), world_coord: ti.types.ndarray()):
    height = depth.shape[0]
    width = depth.shape[1]

    for y, x in ti.ndrange(height, width):
        u = (x + 0.5) / width
        v = (y + 0.5) / height
        lon = (u - 0.5) * 2 * ti.math.pi
        lat = (0.5 - v) * ti.math.pi

        i = y * width + x
        world_coord[0, i] = ti.cos(lat) * ti.sin(lon) * depth[y, x]
        world_coord[1, i] = -ti.sin(lat) * depth[y, x]
        world_coord[2, i] = ti.cos(lat) * ti.cos(lon) * depth[y, x]


def unproject_erp(depth: np.ndarray) -> ti.Ndarray:
    """ERP image pixels → 3D coordinates.

    Args:
        depth: src ERP depth map (radius), shape (H, W).

    Returns:
        3D coordinates, shape (3, H * W).
    """
    height, width = depth.shape

    world_coord = ti.ndarray(ti.f32, shape=(3, height * width))
    unproject_erp_kernel(as_array(depth, np.float32), world_coord)

    return world_coord


@ti.kernel
def project_plane_kernel(world_coord: ti.types.ndarray(), valid_idx: ti.types.ndarray(), x: ti.types.ndarray(), y: ti.types.ndarray(), depth: ti.types.ndarray(), fx: ti.f32, fy: ti.f32, ox: ti.f32, oy: ti.f32, height: ti.i32, width: ti.i32, min_depth: ti.f32):
    for i in range(world_coord.shape[1]):
        valid_idx[i] = -1
        x[i] = -1
        y[i] = -1
        depth[i] = 0.0

        # v - v is not 0 for inf and nan
        finite = (world_coord[0, i] - world_coord[0, i] == 0) and (world_coord[1, i] - world_coord[1, i] == 0) and (world_coord[2, i] - world_coord[2, i] == 0)

        if finite and world_coord[2, i] > min_depth:
            xi = ti.round(world_coord[0, i] * fx / world_coord[2, i] + ox)
            yi = ti.round(world_coord[1, i] * fy / world_coord[2, i] + oy)

            if xi >= 0 and xi < width and yi >= 0 and yi < height:
                valid_idx[i] = i
                x[i] = ti.cast(xi, ti.i32)
                y[i] = ti.cast(yi, ti.i32)
                depth[i] = world_coord[2, i]


def project_plane(world_coord: ti.Ndarray, camera: np.ndarray, height: int, width: int, min_depth: float) -> tuple[ti.Ndarray, ti.Ndarray, ti.Ndarray, ti.Ndarray]:
    """3D coordinates → plane (pinhole) pixel coordinates.

    Points that are non-finite, not deeper than min_depth, or projected
    outside the image are invalid.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        camera: camera intrinsic matrix, shape (3, 3).
        height: dst image height.
        width: dst image width.
        min_depth: lower bound of z (exclusive).

    Returns:
        valid_idx: index of each point in the N input points, shape (N,). -1 for invalid points.
        x: pixel x coordinates (int), shape (N,). -1 for invalid points.
        y: pixel y coordinates (int), shape (N,). -1 for invalid points.
        depth: depth (z) of each point, shape (N,). 0 for invalid points.
    """
    world_coord = as_array(world_coord, np.float32)
    n = world_coord.shape[1]

    valid_idx = ti.ndarray(ti.i32, shape=(n,))
    x = ti.ndarray(ti.i32, shape=(n,))
    y = ti.ndarray(ti.i32, shape=(n,))
    depth = ti.ndarray(ti.f32, shape=(n,))
    project_plane_kernel(world_coord, valid_idx, x, y, depth, camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2], height, width, min_depth)

    return valid_idx, x, y, depth


@ti.kernel
def project_erp_kernel(world_coord: ti.types.ndarray(), valid_idx: ti.types.ndarray(), x: ti.types.ndarray(), y: ti.types.ndarray(), depth: ti.types.ndarray(), height: ti.i32, width: ti.i32):
    for i in range(world_coord.shape[1]):
        valid_idx[i] = -1
        x[i] = -1
        y[i] = -1
        depth[i] = 0.0

        radius = ti.sqrt(world_coord[0, i] ** 2 + world_coord[1, i] ** 2 + world_coord[2, i] ** 2)

        if (radius - radius == 0) and radius > 0:
            lon = ti.atan2(world_coord[0, i], world_coord[2, i])
            lat = ti.asin(ti.min(ti.max(-world_coord[1, i] / radius, -1.0), 1.0))

            xi = ti.cast(ti.round((lon / (2 * ti.math.pi) + 0.5) * width - 0.5), ti.i32)
            yi = ti.cast(ti.round((0.5 - lat / ti.math.pi) * height - 0.5), ti.i32)

            valid_idx[i] = i
            x[i] = (xi % width + width) % width
            y[i] = ti.min(ti.max(yi, 0), height - 1)
            depth[i] = radius


def project_erp(world_coord: ti.Ndarray, height: int, width: int) -> tuple[ti.Ndarray, ti.Ndarray, ti.Ndarray, ti.Ndarray]:
    """3D coordinates → ERP pixel coordinates.

    Points with a non-finite or zero radius are invalid. x is wrapped and
    y is clipped, so every valid point lands inside the image.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        height: ERP image height.
        width: ERP image width.

    Returns:
        valid_idx: index of each point in the N input points, shape (N,). -1 for invalid points.
        x: pixel x coordinates (wrapped, int), shape (N,). -1 for invalid points.
        y: pixel y coordinates (clipped, int), shape (N,). -1 for invalid points.
        depth: depth (radius) of each point, shape (N,). 0 for invalid points.
    """
    world_coord = as_array(world_coord, np.float32)
    n = world_coord.shape[1]

    valid_idx = ti.ndarray(ti.i32, shape=(n,))
    x = ti.ndarray(ti.i32, shape=(n,))
    y = ti.ndarray(ti.i32, shape=(n,))
    depth = ti.ndarray(ti.f32, shape=(n,))
    project_erp_kernel(world_coord, valid_idx, x, y, depth, height, width)

    return valid_idx, x, y, depth


@ti.kernel
def render_image_forward_kernel(ref_image: ti.types.ndarray(), x: ti.types.ndarray(), y: ti.types.ndarray(), src_idx: ti.types.ndarray(), image: ti.types.ndarray()):
    src_width = ref_image.shape[1]

    for i in range(x.shape[0]):
        if src_idx[i] >= 0:
            for c in range(image.shape[2]):
                image[y[i], x[i], c] = ref_image[src_idx[i] // src_width, src_idx[i] % src_width, c]


def render_image_forward(ref_image: np.ndarray, x: ti.Ndarray, y: ti.Ndarray, src_idx: ti.Ndarray, height: int, width: int) -> np.ndarray:
    """Write src colors to dst pixels.

    Args:
        ref_image: src image, shape (H, W, C).
        x: dst pixel x coordinates, shape (N,).
        y: dst pixel y coordinates, shape (N,).
        src_idx: flat src pixel index of each point, shape (N,). Points with -1 are skipped.
        height: dst image height.
        width: dst image width.

    Returns:
        dst image, shape (height, width, C).
    """
    channels = ref_image.shape[2]

    image = ti.ndarray(ti_dtype(ref_image), shape=(height, width, channels))
    image.fill(0)
    render_image_forward_kernel(np.ascontiguousarray(ref_image), x, y, src_idx, image)

    return image.to_numpy()


@ti.kernel
def render_depth_forward_kernel(depth: ti.types.ndarray(), x: ti.types.ndarray(), y: ti.types.ndarray(), depth_map: ti.types.ndarray()):
    for i in range(x.shape[0]):
        if x[i] >= 0:
            depth_map[y[i], x[i]] = depth[i]


def render_depth_forward(depth: ti.Ndarray, x: ti.Ndarray, y: ti.Ndarray, height: int, width: int) -> np.ndarray:
    """Write depths to dst pixels.

    Args:
        depth: depth of each point, shape (N,).
        x: dst pixel x coordinates, shape (N,). Points with -1 are skipped.
        y: dst pixel y coordinates, shape (N,).
        height: dst depth map height.
        width: dst depth map width.

    Returns:
        dst depth map, shape (height, width).
    """
    depth_map = ti.ndarray(ti.f32, shape=(height, width))
    depth_map.fill(0)
    render_depth_forward_kernel(depth, x, y, depth_map)

    return depth_map.to_numpy()


@ti.kernel
def render_image_inverse_kernel(ref_image: ti.types.ndarray(), x: ti.types.ndarray(), y: ti.types.ndarray(), idx: ti.types.ndarray(), image: ti.types.ndarray()):
    width = image.shape[1]

    for i in range(x.shape[0]):
        if idx[i] >= 0:
            for c in range(image.shape[2]):
                image[idx[i] // width, idx[i] % width, c] = ref_image[y[i], x[i], c]


def render_image_inverse(ref_image: np.ndarray, x: ti.Ndarray, y: ti.Ndarray, idx: ti.Ndarray, height: int, width: int) -> np.ndarray:
    """Sample src colors at (x, y) and write them to dst pixels.

    Args:
        ref_image: src image, shape (H, W, C).
        x: src pixel x coordinates, shape (N,).
        y: src pixel y coordinates, shape (N,).
        idx: flat dst pixel index of each point, shape (N,). Points with -1 are skipped.
        height: dst image height.
        width: dst image width.

    Returns:
        dst image, shape (height, width, C).
    """
    channels = ref_image.shape[2]

    image = ti.ndarray(ti_dtype(ref_image), shape=(height, width, channels))
    image.fill(0)
    render_image_inverse_kernel(np.ascontiguousarray(ref_image), x, y, idx, image)

    return image.to_numpy()


@ti.kernel
def apply_rt_kernel(world_coord: ti.types.ndarray(), world_coord_rt: ti.types.ndarray(), R: ti.types.ndarray(), t: ti.types.ndarray()):
    for i in range(world_coord.shape[1]):
        for r in ti.static(range(3)):
            world_coord_rt[r, i] = R[r, 0] * (world_coord[0, i] - t[0, 0]) + R[r, 1] * (world_coord[1, i] - t[1, 0]) + R[r, 2] * (world_coord[2, i] - t[2, 0])


def apply_rt(world_coord: ti.Ndarray, R: np.ndarray, t: np.ndarray) -> ti.Ndarray:
    """Apply R|t to 3D coordinates.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).

    Returns:
        Transformed 3D coordinates, shape (3, N).
    """
    world_coord = as_array(world_coord, np.float32)

    world_coord_rt = ti.ndarray(ti.f32, shape=world_coord.shape)
    apply_rt_kernel(world_coord, world_coord_rt, as_array(R, np.float32), as_array(t, np.float32))

    return world_coord_rt


@ti.kernel
def apply_rt_inverse_kernel(world_coord: ti.types.ndarray(), world_coord_rt: ti.types.ndarray(), R: ti.types.ndarray(), t: ti.types.ndarray()):
    for i in range(world_coord.shape[1]):
        for r in ti.static(range(3)):
            world_coord_rt[r, i] = R[0, r] * world_coord[0, i] + R[1, r] * world_coord[1, i] + R[2, r] * world_coord[2, i] + t[r, 0]


def apply_rt_inverse(world_coord: ti.Ndarray, R: np.ndarray, t: np.ndarray) -> ti.Ndarray:
    """Apply inverse R|t to 3D coordinates.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).

    Returns:
        Transformed 3D coordinates, shape (3, N).
    """
    world_coord = as_array(world_coord, np.float32)

    world_coord_rt = ti.ndarray(ti.f32, shape=world_coord.shape)
    apply_rt_inverse_kernel(world_coord, world_coord_rt, as_array(R, np.float32), as_array(t, np.float32))

    return world_coord_rt


@ti.kernel
def median_kernel(depth: ti.types.ndarray(), depth_filt: ti.types.ndarray()):
    height = depth.shape[0]
    width = depth.shape[1]

    for y, x in ti.ndrange(height, width):
        v = ti.Vector.zero(ti.f32, 25)
        for k in ti.static(range(25)):
            v[k] = depth[ti.min(ti.max(y + k // 5 - 2, 0), height - 1), ti.min(ti.max(x + k % 5 - 2, 0), width - 1)]

        # The median is the value that has 12 smaller and 12 larger values
        result = v[0]
        for i in ti.static(range(25)):
            less = 0
            less_equal = 0
            for j in ti.static(range(25)):
                less += v[j] < v[i]
                less_equal += v[j] <= v[i]
            if less <= 12 and less_equal >= 13:
                result = v[i]

        depth_filt[y, x] = result


@ti.kernel
def morph_kernel(depth: ti.types.ndarray(), depth_filt: ti.types.ndarray(), offsets: ti.types.ndarray(), dilate: ti.i32):
    height = depth.shape[0]
    width = depth.shape[1]

    for y, x in ti.ndrange(height, width):
        value = depth[y, x]
        for k in range(offsets.shape[0]):
            yy = y + offsets[k, 0]
            xx = x + offsets[k, 1]
            if yy >= 0 and yy < height and xx >= 0 and xx < width:
                if dilate == 1:
                    value = ti.max(value, depth[yy, xx])
                else:
                    value = ti.min(value, depth[yy, xx])

        depth_filt[y, x] = value


def morph(depth: ti.Ndarray, size: int, dilate: bool) -> ti.Ndarray:
    """Dilate or erode a depth map with an ellipse structuring element."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    offsets = np.stack(np.nonzero(se), axis=1).astype(np.int32) - size // 2

    depth_filt = ti.ndarray(ti.f32, shape=depth.shape)
    morph_kernel(depth, depth_filt, offsets, int(dilate))

    return depth_filt


def filter_depth(depth: np.ndarray, filter_type: Literal["morph", "median"]) -> np.ndarray:
    """Fill cracks in a warped depth map.

    Args:
        depth: depth map, shape (H, W).
        filter_type: Filter type ("morph" or "median").

    Returns:
        Filtered depth map, shape (H, W).
    """
    if filter_type == "morph":
        depth = morph(as_array(depth, np.float32), 13, True)
        depth = morph(depth, 13, False)
        depth = morph(depth, 5, False)
        depth = depth.to_numpy()
    elif filter_type == "median":
        depth_filt = ti.ndarray(ti.f32, shape=depth.shape)
        median_kernel(as_array(depth, np.float32), depth_filt)
        depth = depth_filt.to_numpy()

    return depth
