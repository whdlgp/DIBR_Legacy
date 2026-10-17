from pathlib import Path
import numpy as np

from dummy_panorama_tool.dummy_panorama_reader import DummyPanoramaReader


if __name__ == "__main__":
    reader = DummyPanoramaReader(Path("dummy_panorama"))
    reader.print_info()

    assert reader.im.dtype == np.uint8 and reader.im.ndim == 3
    assert reader.depth.dtype == np.float32
    assert reader.im.shape[:2] == reader.depth.shape
    assert reader.width == 2 * reader.height
    assert (reader.depth > 0).all()

    reader.resize(0.5)
    assert reader.im.shape[:2] == reader.depth.shape == (reader.height, reader.width)
    print("resize OK:", reader.width, "x", reader.height)

    reader.show_data(800)