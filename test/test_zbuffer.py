import time
import numpy as np


def zbuffer_keep_nearest(x, y, depth, src_idx, height, width):
    target_idx = y.astype(np.int64) * width + x

    zbuf = np.full(height * width, np.inf, dtype=depth.dtype)
    np.minimum.at(zbuf, target_idx, depth)

    winner = depth == zbuf[target_idx]

    return x[winner], y[winner], depth[winner], src_idx[winner]


def zbuffer_keep_nearest_old(x, y, depth, src_idx, height, width):
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

    return x, y, depth, src_idx


def make_points(height, width, count, seed=0):
    rng = np.random.default_rng(seed)

    x = rng.integers(0, width, count).astype(np.int32)
    y = rng.integers(0, height, count).astype(np.int32)
    depth = rng.uniform(0.5, 50.0, count).astype(np.float32)
    src_idx = np.arange(count)

    return x, y, depth, src_idx


def to_depth_map(x, y, depth, height, width):
    depth_map = np.zeros((height, width), dtype=np.float32)
    depth_map[y, x] = depth

    return depth_map


def bench(fn, points, height, width, repeat):
    fn(*points, height, width)

    times = []

    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(*points, height, width)
        times.append(time.perf_counter() - start)

    return result, min(times)


if __name__ == "__main__":
    repeat = 5

    resolutions = [(480, 640), (1080, 1920), (2160, 3840)]
    point_ratios = [0.5, 1.0, 2.0]

    print(f"numpy {np.__version__}")
    print(f"{'resolution':>12} {'points':>10} {'old(s)':>8} {'new(s)':>8} {'speedup':>8} {'match':>6}")

    for height, width in resolutions:
        for ratio in point_ratios:
            count = int(height * width * ratio)
            points = make_points(height, width, count)

            (x_old, y_old, d_old, _), t_old = bench(zbuffer_keep_nearest_old, points, height, width, repeat)
            (x_new, y_new, d_new, _), t_new = bench(zbuffer_keep_nearest, points, height, width, repeat)

            match = np.array_equal(
                to_depth_map(x_old, y_old, d_old, height, width),
                to_depth_map(x_new, y_new, d_new, height, width),
            )

            print(f"{f'{width}x{height}':>12} {count:>10} {t_old:>8.3f} {t_new:>8.3f} {t_old / t_new:>7.2f}x {str(match):>6}")