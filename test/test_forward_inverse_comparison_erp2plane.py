import numpy as np
from pathlib import Path

from dummy_panorama_tool.dummy_panorama_reader import DummyPanoramaReader
from util.dibr import make_rt, forward_mapping_erp2plane, inverse_mapping_erp2plane, show_mapping_comparison


if __name__ == "__main__":
    src = DummyPanoramaReader(Path("dummy_panorama"))

    R, t = make_rt(0, 0, 0, 1000, 0, 0)

    image = src.im
    depth = src.depth

    camera_dst = np.array([
        [600.0,   0.0, 399.5],
        [  0.0, 600.0, 299.5],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float32)
    camera_dst_size = (800, 600)


    print("Test forward mapping")
    image_forward, depth_forward = forward_mapping_erp2plane(depth, image, camera_dst, camera_dst_size, R, t)
    print("Test inverse mapping")
    image_inverse, depth_inverse = inverse_mapping_erp2plane(depth, image, camera_dst, camera_dst_size, R, t)

    show_mapping_comparison([image_forward, image_inverse],
                            [depth_forward, depth_inverse],
                            show_width=900, axis='horizontal')
