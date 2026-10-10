import numpy as np
from pathlib import Path
from util.stereo_reader import StereoReader
from util.dibr import make_rt, inverse_mapping, show_mapping_comparison


if __name__ == "__main__":
    dataset_dir = Path("middlebury2014")
    data_dir = dataset_dir / "Bicycle1-perfect"

    stereo = StereoReader(data_dir)

    R, t = make_rt(0, 0, 0, stereo.calib["baseline"], 0, 0)
    im1_warped, depth1_warped = inverse_mapping(stereo.depth0, stereo.im0, stereo.cam0, stereo.cam1, R, t, verbose=True)

    show_mapping_comparison([stereo.im1, im1_warped],
                            [stereo.depth1, depth1_warped])