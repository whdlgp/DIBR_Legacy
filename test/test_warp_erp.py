from pathlib import Path
from dummy_panorama_tool.dummy_panorama_reader import DummyPanoramaReader
from dibr import make_rt, forward_mapping_erp, inverse_mapping_erp, show_mapping_comparison, set_device


if __name__ == "__main__":
    # Set device
    set_device("auto")
    
    src = DummyPanoramaReader(Path("dummy_panorama"))

    R, t = make_rt(0, 0, 0, 1000, 0, 0)
    print("Test forward mapping")
    im_forward, depth_forward = forward_mapping_erp(src.depth, src.im, R, t, verbose=False)
    print("Test inverse mapping")
    im_inverse, depth_inverse = inverse_mapping_erp(src.depth, src.im, R, t, verbose=False)
    
    show_mapping_comparison([src.im, im_forward, im_inverse],
                            [src.depth, depth_forward, depth_inverse],
                            axis='horizontal')
