import logging
import requests
import gdown
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: Path) -> None:
    """Download a file from a URL to the destination path."""
    logger.info(f"Downloading {url} to {dest_path}")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
    logger.info(f"Downloaded {dest_path}")


def download_models(base_dir: Path = Path("models")) -> None:
    """Download all required models and checkpoints into the specified base directory."""
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download YOLOv8n.pt
    download_file(
        url="https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt",
        dest_path=base_dir / "yolov8n.pt",
    )

    # 2. Download MediaPipe models
    mediapipe_models = {
        "face_landmarker_v2_with_blendshapes.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    }

    for filename, url in mediapipe_models.items():
        download_file(url=url, dest_path=base_dir / filename)

    # 3. Download Google Drive folder
    gdrive_folder_id = "1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV"
    gdrive_output_dir = base_dir / "SHuBERT_ckpts"
    gdrive_output_dir.mkdir(exist_ok=True)

    logger.info(f"Downloading Google Drive folder to {gdrive_output_dir}...")
    gdown.download_folder(
        id=gdrive_folder_id,
        output=str(gdrive_output_dir),
        quiet=False,
        use_cookies=False,
    )

    # 4. Rename specific files
    rename_mapping = {
        "hands_dinov2_checkpoint.pth": "dinov2hand.pth",
        "face_dinov2_checkpoint.pth": "dinov2face.pth",
    }

    for old_name, new_name in rename_mapping.items():
        old_path = gdrive_output_dir / old_name
        new_path = gdrive_output_dir / new_name
        if old_path.exists():
            old_path.rename(new_path)
            logger.info(f"Renamed {old_name} to {new_name}")
        else:
            logger.warning(f"Expected file {old_name} not found, skipping rename.")

    logger.info(f"All downloads and renames completed successfully to {base_dir}")


def main():
    download_models()


if __name__ == "__main__":
    main()
