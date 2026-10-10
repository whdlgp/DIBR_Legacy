from pathlib import Path
from util.stereo_reader import StereoReader


def test():
    dataset_dir = Path("middlebury2014")
    data_dir = dataset_dir / "Bicycle1-perfect"

    stereo_data = StereoReader(data_dir)
    stereo_data.print_info()
    stereo_data.show_data()


if __name__ == "__main__":
    test()