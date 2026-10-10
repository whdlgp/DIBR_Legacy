import cv2
import numpy as np
from pathlib import Path

from util.show import depth_to_show, disp_to_show, show

class StereoReader:
    def __init__(self, data_dir: Path):
        self.data_path = str(data_dir)

        # Left/Right images
        self.im0 = cv2.imread(data_dir / "im0.png")
        self.im1 = cv2.imread(data_dir / "im1.png")

        # width/height info
        self.height, self.width, _ = self.im0.shape

        # Read calib.txt
        self.calib = self._read_calib(data_dir / "calib.txt")

        # Camera intrinsics
        self.cam0 = self.calib["cam0"]
        self.cam1 = self.calib["cam1"]

        # Left/Right disparity
        self.disp0 = self._read_pfm(data_dir / "disp0.pfm").astype(np.float32)
        self.disp1 = self._read_pfm(data_dir / "disp1.pfm").astype(np.float32)

        # Left/Right Depth
        self.depth0 = (self.calib["baseline"] * self.cam0[0, 0] / (self.disp0 + self.calib["doffs"])).astype(np.float32)
        self.depth1 = (self.calib["baseline"] * self.cam1[0, 0] / (self.disp1 + self.calib["doffs"])).astype(np.float32)

    def resize(self, scale):
        # Resize image, disparity, depth and camera parameter
        if scale <= 0:
            raise ValueError("scale must be positive")

        width = round(self.width * scale)
        height = round(self.height * scale)
        size = (width, height)

        self.im0 = cv2.resize(self.im0, size, interpolation=cv2.INTER_AREA)
        self.im1 = cv2.resize(self.im1, size, interpolation=cv2.INTER_AREA)

        self.disp0 = cv2.resize(self.disp0, size, interpolation=cv2.INTER_NEAREST) * scale
        self.disp1 = cv2.resize(self.disp1, size, interpolation=cv2.INTER_NEAREST) * scale

        self.depth0 = cv2.resize(self.depth0, size, interpolation=cv2.INTER_NEAREST)
        self.depth1 = cv2.resize(self.depth1, size, interpolation=cv2.INTER_NEAREST)

        self.cam0[:2, :] *= scale
        self.cam1[:2, :] *= scale

        self.width = width
        self.height = height

    def print_info(self):
        print(f"Path of data: {self.data_path}")
        print(f"Resolution (WxH): {self.width} x {self.height}")

        print(f"Calibration info")
        for key, value in self.calib.items():
            print(f"  {key}: {value}")

        print("Disparity info")
        disp0 = self.disp0[np.isfinite(self.disp0)]
        disp1 = self.disp1[np.isfinite(self.disp1)]
        print(f"disp0: shape={self.disp0.shape}, dtype={self.disp0.dtype}")
        print(f"disp1: shape={self.disp1.shape}, dtype={self.disp1.dtype}")
        print(f"disp0 range: {disp0.min()} ~ {disp0.max()}")
        print(f"disp1 range: {disp1.min()} ~ {disp1.max()}")

        print("Depth info")
        depth0 = self.depth0[np.isfinite(self.depth0) & (self.depth0 > 0)]
        depth1 = self.depth1[np.isfinite(self.depth1) & (self.depth1 > 0)]
        print(f"depth0: shape={self.depth0.shape}, dtype={self.depth0.dtype}")
        print(f"depth1: shape={self.depth1.shape}, dtype={self.depth1.dtype}")
        print(f"depth0 range: {depth0.min()} ~ {depth0.max()}")
        print(f"depth1 range: {depth1.min()} ~ {depth1.max()}")

    def show_data(self, show_width=1280):
        images = cv2.vconcat([self.im0, self.im1])
        disp = cv2.vconcat([self.disp0, self.disp1])
        depth = cv2.vconcat([self.depth0, self.depth1])

        disp_show = disp_to_show(disp)
        depth_show = depth_to_show(depth)

        show_img = cv2.hconcat([images, disp_show, depth_show])
        show(show_img, show_width, "Stereo Data")

    def _read_calib(self, path):
        calib = {}

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1]
                    rows = value.split(";")
                    value = np.array([[float(x) for x in row.split()] for row in rows])
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        value = float(value)

                calib[key] = value

        return calib

    def _read_pfm(self, path):
        with open(path, "rb") as f:
            header = f.readline().decode().strip()
            color = header == "PF"

            if header not in ("PF", "Pf"):
                raise ValueError(f"Invalid PFM header: {header}")

            width, height = map(int, f.readline().decode().split())
            scale = float(f.readline().decode())

            dtype = "<f4" if scale < 0 else ">f4"
            data = np.fromfile(f, dtype=dtype)

            channels = 3 if color else 1
            data = data.reshape((height, width, channels))

            if not color:
                data = data[:, :, 0]

            data = np.flipud(data)

            return data
