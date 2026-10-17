from pathlib import Path

from dibr import inverse_mapping, make_rt, set_device
from util.stereo_reader import StereoReader
from util.visualizer import AnimationVisualizer


if __name__ == "__main__":
    # DIBR input data
    #data_dir = Path("middlebury2014/Bicycle1-perfect")
    data_dir = Path("middlebury2014/Playroom-perfect")
    output_path = Path("output/single_camera_dibr.webp")
    # Resize ratio of input data
    output_width = 960
    # Animation Visualizer parameter
    fps = 20
    duration = 6
    info_height = 76
    move_ratio = 0.2
    corner_power = 4.0
    webp_quality = 80

    # Set device
    set_device("auto")

    # Read input data and resize
    stereo = StereoReader(data_dir)
    stereo.resize(min(1.0, output_width / stereo.width))

    # DIBR callback for visualizer
    def render_function(tx, ty):
        R, t = make_rt(0, 0, 0, tx, ty, 0)
        result, _ = inverse_mapping(stereo.depth0, stereo.im0, stereo.cam0, stereo.cam0, R, t)
        return {"DIBR RESULT": result}

    # Make visualized video(webp)
    visualizer = AnimationVisualizer(render_function, output_path, (stereo.width, stereo.height), fps=fps, duration=duration, move_ratio=move_ratio)
    visualizer.save()