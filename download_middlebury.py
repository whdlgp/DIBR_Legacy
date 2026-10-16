from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skip: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        print(f"Download: {url}")

        with urlopen(url, timeout=60) as response:
            with temp_path.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)

        temp_path.replace(output_path)
        print(f"Saved: {output_path}")

    except (URLError, OSError) as error:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}") from error


if __name__ == "__main__":
    base_url = "https://vision.middlebury.edu/stereo/data/scenes2014/datasets"
    download_dir = Path("middlebury2014")

    scenes = [
        "Bicycle1-perfect",
        "Playroom-perfect",
    ]

    files = [
        "calib.txt",
        "im0.png",
        "im1.png",
        "disp0.pfm",
        "disp1.pfm",
    ]

    for scene in scenes:
        scene_dir = download_dir / scene
        for filename in files:
            url = f"{base_url}/{scene}/{filename}"
            output_path = scene_dir / filename
            download_file(url, output_path)