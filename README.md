# DIBR Legacy

A Python reimplementation of my legacy MATLAB DIBR (Depth Image-Based Rendering) project.

My original MATLAB implementation: [matlab_dibr_test](https://github.com/whdlgp/matlab_dibr_test)

![preview_plane_DIBR](demo_output/preview/stereo_camera_dibr.webp)
![preview_spherical_DIBR](demo_output/preview/erp_camera_walk.webp)

## Dataset

### Plane image (Perspective projection)

Download the Middlebury Stereo 2014 dataset from the [official website](https://vision.middlebury.edu/stereo/data/scenes2014/).

Place the downloaded files in the following directory structure:

```text
middlebury2014/
├── Bicycle1-perfect/
│   ├── calib.txt
│   ├── im0.png
│   ├── im1.png
│   ├── disp0.pfm
│   └── disp1.pfm
└── Playroom-perfect/
    ├── calib.txt
    ├── im0.png
    ├── im1.png
    ├── disp0.pfm
    └── disp1.pfm
```

Dataset reference:  
Scharstein et al., *High-resolution stereo datasets with subpixel-accurate ground truth*, GCPR 2014.

### Spherical image (ERP, Equirectangular projection)

Most ERP datasets with depth GT require license agreements and access requests.  
I understand why they ask for it, but it's too much work just to get a simple test input.  
So I built my own test input generator.

```bash
uv run python -m dummy_panorama_tool.dummy_panorama_creater
```

<details>
<summary>Dummy Panorama image example</summary>

![spherical_dibr](demo_output/spherical_dibr/room_rgb.png)

</details>

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

Single-camera virtual viewpoint rendering demo:

```bash
uv run demo_single_camera.py
```

Stereo-camera midpoint rendering demo:

```bash
uv run demo_stereo_camera.py
```

Spherical(ERP, Equirectangular projection) to Plane viewpoint rendering demo:

```bash
uv run demo_erp2plane.py
```

## Result

Animated WebP demos.

<details>
<summary>Single-camera virtual viewpoint rendering</summary>

![Bicycle1](demo_output/Bicycle1/single_camera_dibr.webp)

![Playroom](demo_output/Playroom/single_camera_dibr.webp)

</details>

<details>
<summary>Stereo-camera midpoint rendering</summary>

![Bicycle1](demo_output/Bicycle1/stereo_camera_dibr.webp)

![Playroom](demo_output/Playroom/stereo_camera_dibr.webp)

</details>

<details>
<summary>Spherical(ERP, Equirectangular projection) to Plane viewpoint rendering</summary>

![spherical_dibr](demo_output/spherical_dibr/erp_camera_walk.webp)

</details>

## License

The source code is licensed under the MIT License.

Demo outputs are generated using the Middlebury Stereo 2014 dataset
and are not covered by the source code license.

Dataset: [Middlebury Stereo 2014](https://vision.middlebury.edu/stereo/data/scenes2014/)