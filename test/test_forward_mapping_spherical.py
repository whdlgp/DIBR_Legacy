from pathlib import Path
from dummy_panorama_tool.dummy_panorama_reader import DummyPanoramaReader
from dibr import make_rt, forward_mapping_erp, show_mapping_comparison


if __name__ == "__main__":
    src = DummyPanoramaReader(Path("dummy_panorama"))

    R, t = make_rt(0, 0, 0, 1000, 0, 0)
    im_warped, depth_warped = forward_mapping_erp(src.depth, src.im, R, t, verbose=True)

    show_mapping_comparison([src.im, im_warped],
                            [src.depth, depth_warped])