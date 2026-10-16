import numpy as np
from pathlib import Path
from util.stereo_reader import StereoReader
from dibr import make_rt, inverse_mapping, alpha_blending
from util.show import show

if __name__ == "__main__":
    dataset_dir = Path("middlebury2014")
    data_dir = dataset_dir / "Bicycle1-perfect"

    stereo = StereoReader(data_dir)

    print("---Set up virtual camera at the midpoint of cameras---")
    baseline = stereo.calib["baseline"]
    R_left, t_left = make_rt(0, 0, 0, baseline / 2, 0, 0)
    R_right, t_right = make_rt(0, 0, 0, -baseline / 2, 0, 0)

    print("---Map left image to center---")
    im_left, depth_left = inverse_mapping(stereo.depth0, stereo.im0, stereo.cam0, stereo.cam0, R_left, t_left, verbose=True)

    print("---Map right image to center---")
    im_right, depth_right = inverse_mapping(stereo.depth1, stereo.im1, stereo.cam1, stereo.cam0, R_right, t_right, verbose=True)

    print("---Calculate alpha---")
    norm = np.linalg.norm
    t_l = np.array([0, 0, 0])
    t_r = np.array([baseline, 0, 0])
    t_virtual = np.array([baseline / 2, 0, 0])
    alpha = norm(t_virtual - t_l) / (norm(t_virtual - t_l) + norm(t_virtual - t_r))

    print("---Alpha blending---")
    im_blended = alpha_blending(im_left, im_right, depth_left, depth_right, alpha)

    show(im_blended, title="Alpha Blending")