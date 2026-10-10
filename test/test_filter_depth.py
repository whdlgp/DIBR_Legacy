import numpy as np
from pathlib import Path
from util.stereo_reader import StereoReader
from util.dibr import make_rt, forward_mapping, filter_depth
from util.show import depth_to_show, show

import cv2

if __name__ == "__main__":
    dataset_dir = Path("middlebury2014")
    data_dir = dataset_dir / "Bicycle1-perfect"

    stereo = StereoReader(data_dir)

    print("Forward mapping depth for test")
    R, t = make_rt(0, 0, 0, stereo.calib["baseline"], 0, 0)
    im1_warped, depth1_warped = forward_mapping(stereo.depth0, stereo.im0, stereo.cam0, stereo.cam1, R, t, 'depth')
    
    print("Morphological filtering")
    depth_morph = filter_depth(depth1_warped, 'morph')

    print("Median filtering")
    depth_median = filter_depth(depth1_warped, 'median')

    print("Done")

    depth_show = cv2.hconcat([depth1_warped, depth_morph, depth_median])
    depth_show = depth_to_show(depth_show)
    show(depth_show, title="Depth Filtering Comparison")