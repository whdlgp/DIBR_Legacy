import numpy as np
from pathlib import Path
from util.stereo_reader import StereoReader
from dibr import make_rt, forward_mapping, inverse_mapping, show_mapping_comparison, set_device


if __name__ == "__main__":
    # Set device
    set_device("auto")
    
    dataset_dir = Path("middlebury2014")
    data_dir = dataset_dir / "Bicycle1-perfect"

    stereo = StereoReader(data_dir)

    R, t = make_rt(0, 0, 0, stereo.calib["baseline"], 0, 0)
    print("Test forward mapping")
    im1_forward, depth1_forward = forward_mapping(stereo.depth0, stereo.im0, stereo.cam0, stereo.cam1, R, t, verbose=False)
    print("Test inverse mapping")
    im1_inverse, depth1_inverse = inverse_mapping(stereo.depth0, stereo.im0, stereo.cam0, stereo.cam1, R, t, verbose=False)

    show_mapping_comparison([stereo.im1, im1_forward, im1_inverse],
                            [stereo.depth1, depth1_forward, depth1_inverse]
                            , axis='horizontal')
