import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time
import math


class AnimationVisualizer:
    def __init__(self, render_function, output_path, image_size, fps=20, duration=6, info_height=76, move_ratio=0.025, corner_power=4.0, webp_quality=80):
        self.render_function = render_function
        self.output_path = Path(output_path)
        self.width, self.height = image_size
        self.fps = fps
        self.duration = duration
        self.info_height = info_height
        self.move_ratio = move_ratio
        self.corner_power = corner_power
        self.webp_quality = webp_quality

    def save(self):
        frame_count = self.fps * self.duration
        half_width = self.width * self.move_ratio
        half_height = self.height * self.move_ratio
        path = self.rounded_square_path(frame_count, half_width, half_height, self.corner_power)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        frames, elapsed_times = [], []

        for frame_index, (tx, ty) in enumerate(tqdm(path, desc="Rendering frames", unit="frame")):
            start = time.perf_counter()
            images = self.render_function(float(tx), float(ty))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            elapsed_times.append(elapsed_ms)
            current_fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
            average_fps = 1000.0 / np.mean(elapsed_times)
            frame = self.draw_frame(images, elapsed_ms, current_fps, average_fps,
                                    tx, ty, frame_index, frame_count)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        print(f"Saving WebP: {self.output_path}")
        frames[0].save(self.output_path, format="WEBP", save_all=True, append_images=frames[1:],
                       duration=1000 // self.fps, loop=0, quality=self.webp_quality, method=4)
        print(f"Saved: {self.output_path}")
        print(f"Average render time: {np.mean(elapsed_times):.2f} ms ({1000.0 / np.mean(elapsed_times):.1f} FPS)")

    @staticmethod
    def rounded_square_path(count, half_width, half_height, power=4.0):
        samples = max(4096, count * 32)
        theta = np.linspace(0, 2 * math.pi, samples + 1)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x = half_width * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / power)
        y = half_height * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / power)
        lengths = np.linalg.norm(np.diff(np.column_stack((x, y)), axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        targets = np.arange(count) * (cumulative[-1] / count)
        return np.column_stack((np.interp(targets, cumulative, x), np.interp(targets, cumulative, y)))

    def draw_frame(self, images, elapsed_ms, fps, average_fps, tx, ty, frame_index, frame_count):
        if isinstance(images, np.ndarray):
            images = [("RESULT", images)]
        elif isinstance(images, dict):
            images = list(images.items())
        else:
            images = [(f"IMAGE {i + 1}", image) for i, image in enumerate(images)]

        panels = []
        for _, image in images:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if image.shape[:2] != (self.height, self.width):
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            panels.append(image)

        content = cv2.hconcat(panels) if len(panels) > 1 else panels[0]
        height, width = content.shape[:2]
        info_height = max(self.info_height, 88)
        canvas = np.zeros((height + info_height, width, 3), dtype=np.uint8)
        canvas[:info_height] = (28, 31, 37)
        canvas[info_height:] = content

        scale = max(0.5, width / 1280)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color, muted = (240, 243, 248), (165, 177, 192)
        title = "  |  ".join(str(name).upper() for name, _ in images)
        cv2.putText(canvas, title, (20, round(25 * scale)), font, 0.58 * scale, color, 1, cv2.LINE_AA)

        info = [
            f"RENDER  {elapsed_ms:.2f} ms",
            f"FPS  {fps:.1f}  /  AVG  {average_fps:.1f}",
            f"CAMERA  ({tx:+.4f}, {ty:+.4f})",
            f"FRAME  {frame_index + 1}/{frame_count}",
        ]
        x = 20
        for label in info:
            cv2.putText(canvas, label, (x, round(62 * scale)), font, 0.42 * scale, muted, 1, cv2.LINE_AA)
            (text_width, _), _ = cv2.getTextSize(label, font, 0.42 * scale, 1)
            x += text_width + round(20 * scale)

        return canvas